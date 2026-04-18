import copy
import math
import os

import torch as th
import torch.nn.functional as F
from torch.optim import Adam

from modules.mixers.soft_mix import SoftMixer
from utils.rl_utils import build_td_lambda_targets


class CASVDLearner:
    """CASVD learner — Soft-QMIX backbone with GAT encoder.

    Architecture (Component 2: Soft-QMIX port):
    - SoftMixer: VDN forward + func_g (order-preserving Q shaping)
      + func_f (learned per-agent temperature for soft policy).
    - Soft policy: softmax(func_f(func_g(Q_online)) / α) for rollout
      and target action sampling.
    - Sample-based entropy estimate added to TD(λ) targets.
    - Beta loss keeps func_f ≈ identity.
    - Weighted TD loss for stability.

    Components 3 (InfoNCE) and 4 (adaptive α) are retained in code
    but gated by lgdd_enabled / use_adaptive_alpha flags (off by default).
    """

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        # Target network
        self.target_mac = copy.deepcopy(self.mac)

        # ── Soft-QMIX mixer: VDN + func_f + func_g ──
        self.mixer = SoftMixer(args)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.mac.set_mixer(self.mixer)

        self.entropy_coef = getattr(args, "entropy_coef", 0.03)

        # ── Per-agent adaptive alpha (off by default) ──
        self.use_adaptive_alpha = getattr(args, "use_adaptive_alpha", False)
        self._coord_signals = None
        self._coord_warmup = True   # first batch seeds directly, no EMA blending
        # τ = 0.95 → time constant ~20 training steps ≈ 0.2 % of a 10 M-step run.
        # Fast enough that early-training coordination changes actually move the
        # signal; slow enough that per-batch sampling noise averages out.
        self.coord_signal_ema_tau = getattr(args, "coord_signal_ema_tau", 0.95)

        # ── InfoNCE coordination sensor (off by default) ──
        self.lgdd_enabled = getattr(args, "lgdd_enabled", False)
        self.infonce_n_negatives = getattr(args, "infonce_n_negatives", 15)
        self.infonce_temperature = getattr(args, "infonce_temperature", 0.1)

        self.dynamics_predictor = None
        self.dynamics_params = []

        if self.lgdd_enabled:
            from modules.predictors import InfoNCEPredictor

            self.dynamics_predictor = InfoNCEPredictor(
                args.hidden_dim,
                temperature=self.infonce_temperature,
            )
            self.dynamics_params = list(self.dynamics_predictor.parameters())

        # ── Continual learning (off by default) ──
        self.cl_enabled = getattr(args, "cl_enabled", False)
        self.cl_distill_weight = getattr(args, "cl_distill_weight", 0.0)
        self.cl_teacher_ema_tau = getattr(args, "cl_teacher_ema_tau", 0.002)
        self.cl_teacher_mac = None

        if self.cl_enabled and self.cl_distill_weight > 0:
            self.cl_teacher_mac = copy.deepcopy(self.mac)
            self._freeze_mac(self.cl_teacher_mac)

        # ── Optimizers ──
        opt_eps = getattr(args, "optimizer_epsilon", 1e-7)
        self.main_params = list(self.mac.parameters()) + list(self.mixer.parameters())
        self.main_optimizer = Adam(self.main_params, lr=args.lr, eps=opt_eps)

        if self.dynamics_params:
            lgdd_lr = getattr(args, "lgdd_lr", args.lr)
            self.lgdd_optimizer = Adam(self.dynamics_params, lr=lgdd_lr, eps=opt_eps)
        else:
            self.lgdd_optimizer = None

        self.params = self.main_params
        self.last_target_update_episode = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

    def _get_coord_signals(self, device):
        """Lazy-init per-agent coordination signal buffer.

        Only creates ones() on the very first call.  Subsequent calls
        return the accumulated EMA tensor, moving it to *device* if the
        device has changed (without resetting values).
        """
        if self._coord_signals is None:
            self._coord_signals = th.ones(self.n_agents, device=device)
        elif self._coord_signals.device != device:
            self._coord_signals = self._coord_signals.to(device)
        return self._coord_signals

    def _compute_infonce(self, all_hidden, mask, device):
        """Per-agent InfoNCE coordination signal with cross-episode negatives.

        The positive sample for agent i at time t is `g_others_i[t+1]`,
        the mean of OTHER agents' hidden states at the next timestep in
        the SAME episode (asymmetric — breaks encoder symmetry).

        The K negative samples are drawn from OTHER episodes in the
        batch.  This is the fix for the "same-episode negative" bug:
        if negatives come from other timesteps of the same episode,
        they share team composition, start positions, and temporally-
        correlated game state, so the predictor collapses to a trivial
        timestamp classifier.  With cross-episode negatives the task
        is "which next-state belongs to my battle" — only genuine
        coordination features can solve it.

        Args:
            all_hidden: list of [B, n_agents, D] hidden tensors, length T.
            mask:       [B, T-1, 1] valid-timestep mask.
            device:     target device.

        Returns:
            infonce_loss:       scalar, mean InfoNCE loss for backward.
            coord_signal_batch: [n_agents] normalised per-agent signal.
            per_agent_loss:     [B, n_agents] time-masked per-agent loss.
        """
        all_h = th.stack(all_hidden, dim=0)           # [T, B, N, D]
        T_size, B_size, N, D = all_h.shape
        K = self.infonce_n_negatives

        # Per-agent positive: mean of OTHER agents' hidden states at t+1
        if N > 1:
            team_sum = all_h.sum(dim=2, keepdim=True)       # [T, B, 1, D]
            g_others = (team_sum - all_h) / (N - 1)         # [T, B, N, D]
        else:
            g_others = all_h                                # degenerate

        # Global team mean (for sampling negatives across the batch)
        g_all = all_h.mean(dim=2)                           # [T, B, D]

        # ── Cross-episode negative sampling (FIX for Problem 1) ─────
        # For every positive at (b, t+1), sample K negatives as
        # (b', t') pairs with b' ≠ b.  Guarantees temporal-shortcut
        # features (elapsed time, cumulative damage, dead-unit count)
        # cannot solve the task — the predictor MUST rely on
        # coordination-relevant features to distinguish the true
        # next-state from B−1 alternative battles' states.
        if B_size >= 2:
            shifts = th.randint(1, B_size, (T_size - 1, B_size, K), device=device)
        else:
            # Degenerate: batch size 1.  Should not happen in normal
            # training (batch_size=128 in the config).
            shifts = th.zeros((T_size - 1, B_size, K), dtype=th.long, device=device)
        neg_t = th.randint(0, T_size, (T_size - 1, B_size, K), device=device)
        b_arange = th.arange(B_size, device=device).view(1, B_size, 1)
        neg_b = (b_arange + shifts) % B_size                # [T-1, B, K], ≠ b

        # Gather: g_all[neg_t[τ,b,k], neg_b[τ,b,k]]  →  [T-1, B, K, D]
        g_neg_all = g_all[neg_t, neg_b]

        # ── Vectorised predictor call ────────────────────────────
        # Flatten (T-1) and B into one big batch dim for a single call.
        h_i_all   = all_h[:T_size - 1]                      # [T-1, B, N, D]
        g_pos_all = g_others[1:T_size]                      # [T-1, B, N, D]

        h_i_flat   = h_i_all.reshape(-1, N, D)              # [(T-1)*B, N, D]
        g_pos_flat = g_pos_all.reshape(-1, N, D)            # [(T-1)*B, N, D]
        g_neg_flat = g_neg_all.reshape(-1, K, D)            # [(T-1)*B, K, D]

        loss_flat, pred_stats = self.dynamics_predictor(
            h_i_flat, g_pos_flat, g_neg_flat, return_stats=True
        )                                                    # [(T-1)*B, N], dict

        all_per_agent_loss = loss_flat.reshape(
            T_size - 1, B_size, N
        ).permute(1, 0, 2)                                   # [B, T-1, N]

        # Masked time-average → [B, N]
        mask_sq  = mask.squeeze(-1)                          # [B, T-1]
        mask_exp = mask_sq.unsqueeze(-1)                     # [B, T-1, 1]
        denom    = mask_exp.sum(dim=1).clamp(min=1.0)
        per_agent_loss = (all_per_agent_loss * mask_exp).sum(dim=1) / denom

        # Scalar loss for backward (mean over B and N)
        infonce_loss = per_agent_loss.mean()

        # Per-agent coordination signal in [0, 1]
        max_infonce = math.log(K + 1)
        coord_signal_batch = (
            per_agent_loss.mean(dim=0) / max_infonce
        ).clamp(0.0, 1.0)

        # ── Additional diagnostic stats ──────────────────────────
        # Captured once per train() call; cheap to compute.
        with th.no_grad():
            # Cross-agent hidden-state diversity (per-timestep std across agents).
            # Low value → clustered formation (h_i's all similar).
            # High value → dispersed/flanking formation (h_i's diverge).
            h_diversity = all_h.std(dim=2).mean().item()

            # Raw cosine similarity of h_i and g_pos BEFORE predictor projection.
            # If already high, predictor can succeed via W ≈ identity → task trivial.
            h_norm = F.normalize(h_i_all, dim=-1)
            g_norm = F.normalize(g_pos_all, dim=-1)
            raw_cos_sim = (h_norm * g_norm).sum(-1).mean().item()

            # Loss distribution percentiles — shows if most samples are easy
            # (loss near 0) or if there's real variance.
            flat_loss = all_per_agent_loss.reshape(-1)
            q = th.quantile(
                flat_loss, th.tensor([0.1, 0.5, 0.9], device=flat_loss.device)
            )
            loss_p10, loss_p50, loss_p90 = q[0].item(), q[1].item(), q[2].item()

        diag = {
            "h_diversity":   h_diversity,
            "raw_cos_sim":   raw_cos_sim,
            "loss_p10":      loss_p10,
            "loss_median":   loss_p50,
            "loss_p90":      loss_p90,
            "top1_acc":      pred_stats["top1_acc_per_agent"].mean().item(),
            "top1_per_agent": pred_stats["top1_acc_per_agent"],       # [N]
            "pos_score":     pred_stats["pos_score_mean"],
            "neg_score":     pred_stats["neg_score_mean"],
            "margin":        pred_stats["margin_mean"],
            "per_agent_loss_mean": per_agent_loss.mean(dim=0),          # [N]
        }

        return infonce_loss, coord_signal_batch, per_agent_loss, diag

    def train(self, batch, t_env, episode_num, current_batch_size=None, memory_batch_size=0):
        # ── Split batch for CL ──
        actual_current = current_batch_size if current_batch_size is not None else batch.batch_size
        if memory_batch_size > 0 and batch.batch_size > actual_current:
            current_batch = batch[:actual_current]
            memory_batch = batch[actual_current:]
        else:
            current_batch = batch
            memory_batch = None

        rewards = current_batch["reward"][:, :-1]
        actions = current_batch["actions"][:, :-1]
        terminated = current_batch["terminated"][:, :-1].float()
        mask = current_batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = current_batch["avail_actions"]
        states = current_batch["state"]

        # ═══════════════════════════════════════════════════════
        # 1. Online forward: raw Q → func_g → mac_out
        # ═══════════════════════════════════════════════════════
        self.mac.init_hidden(current_batch.batch_size)
        mac_out = []
        all_hidden = [] if self.lgdd_enabled else None
        for t in range(current_batch.max_seq_length):
            q_values = self.mac.forward(current_batch, t)
            mac_out.append(q_values)
            if self.lgdd_enabled:
                # Detached clone → InfoNCE gradient never flows into encoder.
                all_hidden.append(self.mac.hidden_states.detach().clone())
        mac_out = th.stack(mac_out, dim=1)
        mac_out = self.mixer.func_g(mac_out, states, t_env)

        chosen_action_qvals = th.gather(
            mac_out[:, :-1], dim=3, index=actions
        ).squeeze(3)

        # ═══════════════════════════════════════════════════════
        # 2. Target: func_g(Q_target) + soft policy sampling
        # ═══════════════════════════════════════════════════════
        with th.no_grad():
            self.target_mac.init_hidden(current_batch.batch_size)
            target_mac_out = []
            for t in range(current_batch.max_seq_length):
                target_q = self.target_mac.forward(current_batch, t)
                target_mac_out.append(target_q)
            target_mac_out = th.stack(target_mac_out, dim=1)
            target_mac_out = self.target_mixer.func_g(target_mac_out, states, t_env)

            # Soft policy from online net (Double-Q: online selects)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach = self.mixer.func_f(mac_out_detach, states, t_env)
            mac_out_detach = mac_out_detach / self.entropy_coef
            mac_out_detach[avail_actions == 0] = -9999999
            actions_pdf = th.softmax(mac_out_detach, dim=-1)

            # Sample actions via CDF trick (numerically stable)
            rand_idx = th.rand(
                actions_pdf[:, :, :, :1].shape, device=actions_pdf.device
            )
            actions_cdf = th.cumsum(actions_pdf, dim=-1)
            rand_idx = th.clamp(rand_idx, 1e-6, 1 - 1e-6)
            picked_actions = th.searchsorted(actions_cdf, rand_idx)

            # Target Q at sampled actions (Double-Q: target evaluates)
            target_qvals = th.gather(
                target_mac_out.clone(), 3, picked_actions
            ).squeeze(3)

            # Single-sample entropy estimate: -Σ_i log π_i(a*_i)
            target_logp = th.log(actions_pdf + 1e-10)
            target_logp = th.gather(target_logp, 3, picked_actions).squeeze(3)
            target_entropy = -target_logp.sum(-1, keepdim=True)

            # Mix sampled target Q through VDN sum
            target_qvals = self.target_mixer(target_qvals, states)

            # TD(λ) with entropy bonus
            targets = build_td_lambda_targets(
                rewards, terminated, mask,
                target_qvals, self.n_agents,
                self.args.gamma, self.args.td_lambda,
                target_entropy=target_entropy * self.entropy_coef,
            )

        # ═══════════════════════════════════════════════════════
        # 3. Mix chosen Q through VDN sum
        # ═══════════════════════════════════════════════════════
        chosen_aq_clone = chosen_action_qvals.clone().detach()
        chosen_action_qvals_mixed = self.mixer(
            chosen_action_qvals, states[:, :-1]
        )

        # ═══════════════════════════════════════════════════════
        # 4. TD loss (standard MSE, for logging)
        # ═══════════════════════════════════════════════════════
        td_error = chosen_action_qvals_mixed - targets.detach()
        td_error_sq = 0.5 * td_error.pow(2)
        mask_exp = mask.expand_as(td_error_sq)
        masked_td_error = td_error_sq * mask_exp
        L_td = masked_td_error.sum() / mask_exp.sum()

        # ═══════════════════════════════════════════════════════
        # 5. Beta loss: keeps func_f ≈ identity
        # ═══════════════════════════════════════════════════════
        affine_aq = self.mixer.func_f(
            chosen_aq_clone, states[:, :-1], t_env
        )
        approx_error = chosen_action_qvals_mixed.detach() - affine_aq.sum(-1, keepdim=True)
        beta_error = 0.5 * approx_error.pow(2)
        masked_beta_error = beta_error * mask_exp
        L_beta = masked_beta_error.sum() / mask_exp.sum()

        # ═══════════════════════════════════════════════════════
        # 6. Weighted TD loss (asymmetric consistency mask)
        # ═══════════════════════════════════════════════════════
        gopt_mask = (
            ((approx_error > 0.0).float() + (td_error < 0.0).float()) != 1
        ).float()
        weight_td = masked_td_error * gopt_mask * 0.5 + masked_td_error * (1 - gopt_mask)
        mask_sum = mask_exp * gopt_mask * 0.5 + mask_exp * (1 - gopt_mask)
        L_wtd = weight_td.sum() / mask_sum.sum()

        loss = L_wtd + L_beta

        # ═══════════════════════════════════════════════════════
        # 7a. InfoNCE coordination signal (gradient-isolated)
        #     Uses CROSS-EPISODE negatives (fix for same-episode
        #     temporal-shortcut bug).  Only the predictor MLP is
        #     updated — hidden states are detached, so no gradient
        #     flows into the encoder/mixer.
        # ═══════════════════════════════════════════════════════
        infonce_loss = None
        coord_signal_mean = 0.0
        infonce_diag = None
        if self.lgdd_enabled and self.dynamics_predictor is not None:
            infonce_loss, coord_signal_batch, _, infonce_diag = self._compute_infonce(
                all_hidden, mask, current_batch.device
            )
            # Warm-start: on the very first batch, seed coord_signals directly
            # from the measured values instead of blending with the ones()
            # initialisation.  Without this, τ=0.95 means the first 60-100
            # batches are still dominated by the prior (~1.0), wasting the
            # early-training window where coordination is changing fastest.
            _ = self._get_coord_signals(current_batch.device)  # lazy-init buffer
            if self._coord_warmup:
                self._coord_signals = coord_signal_batch.detach().clone()
                self._coord_warmup = False
            else:
                tau = self.coord_signal_ema_tau
                self._coord_signals = (
                    tau * self._coord_signals
                    + (1.0 - tau) * coord_signal_batch.detach()
                )
            coord_signal_mean = self._coord_signals.mean().item()

        # ═══════════════════════════════════════════════════════
        # 7b. InfoNCE backward (separate optimiser, isolated graph)
        # ═══════════════════════════════════════════════════════
        if infonce_loss is not None and self.lgdd_optimizer is not None:
            self.lgdd_optimizer.zero_grad()
            infonce_loss.backward()
            th.nn.utils.clip_grad_norm_(self.dynamics_params, self.args.grad_norm_clip)
            self.lgdd_optimizer.step()

        # ═══════════════════════════════════════════════════════
        # 7c. Main network backward + optimise
        # ═══════════════════════════════════════════════════════
        self.main_optimizer.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.main_params, self.args.grad_norm_clip)
        self.main_optimizer.step()

        # ═══════════════════════════════════════════════════════
        # 8. Target network updates
        # ═══════════════════════════════════════════════════════
        tau = self.args.target_update_interval_or_tau
        if tau > 1:
            if (episode_num - self.last_target_update_episode) / tau >= 1.0:
                self._update_targets_hard()
                self.last_target_update_episode = episode_num
        else:
            self._update_targets_soft(tau)

        if self.cl_teacher_mac is not None:
            self._update_teacher_mac()

        # ═══════════════════════════════════════════════════════
        # 9. Logging
        # ═══════════════════════════════════════════════════════
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = mask_exp.sum().item()
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("loss_wtd", L_wtd.item(), t_env)
            self.logger.log_stat("loss_beta", L_beta.item(), t_env)
            self.logger.log_stat(
                "grad_norm",
                grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
                t_env,
            )
            self.logger.log_stat("agent_q_mean", mac_out.mean().item(), t_env)
            self.logger.log_stat("agent_q_std", mac_out.std().item(), t_env)
            self.logger.log_stat("agent_q_max_abs", mac_out.abs().max().item(), t_env)
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals_mixed * mask_exp).sum().item() / mask_elems,
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask_exp).sum().item() / mask_elems,
                t_env,
            )
            self.logger.log_stat("entropy", target_entropy.mean().item(), t_env)
            self.logger.log_stat("entropy_coef", self.entropy_coef, t_env)
            self.logger.log_stat(
                "err_mask", (mask_sum.sum() / mask_exp.sum()).item(), t_env
            )
            if self.lgdd_enabled and infonce_loss is not None:
                # ── Loss & signal stats ──────────────────────────
                self.logger.log_stat("infonce_loss", infonce_loss.item(), t_env)
                self.logger.log_stat("coord_signal_mean", coord_signal_mean, t_env)
                if self._coord_signals is not None:
                    self.logger.log_stat(
                        "coord_signal_std",
                        self._coord_signals.std().item(),
                        t_env,
                    )
                    self.logger.log_stat(
                        "coord_signal_min",
                        self._coord_signals.min().item(),
                        t_env,
                    )
                    self.logger.log_stat(
                        "coord_signal_max",
                        self._coord_signals.max().item(),
                        t_env,
                    )
                    # Per-agent coord signals → see if agents actually differ
                    for i in range(self.n_agents):
                        self.logger.log_stat(
                            f"coord_signal_agent_{i}",
                            self._coord_signals[i].item(),
                            t_env,
                        )

                if infonce_diag is not None:
                    # ── Task difficulty diagnostics ─────────────────
                    #   raw_cos_sim : cosine(h_i, g_pos) without predictor
                    #                 projection. If >0.5, task is trivially
                    #                 solved via W ≈ identity (bad — means
                    #                 coord_signal is measuring self-similarity).
                    #   top1_acc    : fraction of samples where the predictor
                    #                 ranks the positive above all K negatives.
                    #                 Random = 1/(K+1) = 0.0625.  Target: 0.3-0.8.
                    #                 Near 1.0 = task too easy; near 0.06 = too hard.
                    #   margin      : pos_score - max_neg_score (after τ-unscaling).
                    #                 Large positive = predictor very confident;
                    #                 near zero = predictor struggling.
                    self.logger.log_stat(
                        "infonce_raw_cos_sim", infonce_diag["raw_cos_sim"], t_env
                    )
                    self.logger.log_stat(
                        "infonce_top1_acc", infonce_diag["top1_acc"], t_env
                    )
                    self.logger.log_stat(
                        "infonce_margin", infonce_diag["margin"], t_env
                    )
                    self.logger.log_stat(
                        "infonce_pos_score", infonce_diag["pos_score"], t_env
                    )
                    self.logger.log_stat(
                        "infonce_neg_score", infonce_diag["neg_score"], t_env
                    )

                    # ── Loss distribution (percentiles over all (b, t, i) samples)
                    self.logger.log_stat(
                        "infonce_loss_p10", infonce_diag["loss_p10"], t_env
                    )
                    self.logger.log_stat(
                        "infonce_loss_median", infonce_diag["loss_median"], t_env
                    )
                    self.logger.log_stat(
                        "infonce_loss_p90", infonce_diag["loss_p90"], t_env
                    )

                    # ── Formation proxy ─────────────────────────────
                    # Cross-agent hidden-state std.  Low = clustered formation;
                    # high = flanking/dispersed.  Correlate with coord_signal
                    # across episodes to detect the Problem-3 formation bias.
                    self.logger.log_stat(
                        "h_diversity", infonce_diag["h_diversity"], t_env
                    )

                    # ── Per-agent InfoNCE loss and top-1 accuracy ───
                    for i in range(self.n_agents):
                        self.logger.log_stat(
                            f"infonce_loss_agent_{i}",
                            infonce_diag["per_agent_loss_mean"][i].item(),
                            t_env,
                        )
                        self.logger.log_stat(
                            f"infonce_top1_agent_{i}",
                            infonce_diag["top1_per_agent"][i].item(),
                            t_env,
                        )
            self.log_stats_t = t_env

    # ═══════════════════════════════════════════════════════════
    # CL Distillation
    # ═══════════════════════════════════════════════════════════

    def _compute_cl_distillation(self, memory_batch):
        """KL divergence between teacher and student Q-value distributions.

        Uses a fixed temperature for Boltzmann policy comparison,
        independent of the soft value alpha_factor.
        """
        if memory_batch is None or memory_batch.batch_size == 0:
            return th.tensor(0.0, device=self.args.device)

        mem_mask = memory_batch["filled"][:, :-1].float()
        mem_terminated = memory_batch["terminated"][:, :-1].float()
        mem_mask[:, 1:] = mem_mask[:, 1:] * (1 - mem_terminated[:, :-1])
        mem_mask = mem_mask.expand(-1, -1, self.n_agents)
        avail = memory_batch["avail_actions"][:, :-1]

        # Fixed temperature for CL distillation (independent of soft value α)
        cl_temperature = getattr(self.args, "cl_temperature", 0.1)

        # Student Q-values -> Boltzmann policy
        self.mac.init_hidden(memory_batch.batch_size)
        student_qs = []
        for t in range(memory_batch.max_seq_length - 1):
            q_t = self.mac.forward(memory_batch, t)
            student_qs.append(q_t)
        student_qs = th.stack(student_qs, dim=1)  # [B, T-1, n_agents, n_actions]
        student_qs[avail == 0] = -1e10
        student_probs = th.softmax(student_qs / cl_temperature, dim=-1)

        # Teacher Q-values -> Boltzmann policy
        with th.no_grad():
            self.cl_teacher_mac.init_hidden(memory_batch.batch_size)
            teacher_qs = []
            for t in range(memory_batch.max_seq_length - 1):
                q_t = self.cl_teacher_mac.forward(memory_batch, t)
                teacher_qs.append(q_t)
            teacher_qs = th.stack(teacher_qs, dim=1)
            teacher_qs[avail == 0] = -1e10
            teacher_probs = th.softmax(teacher_qs / cl_temperature, dim=-1)

        # KL(teacher || student)
        kl = teacher_probs * (th.log(teacher_probs + 1e-10) - th.log(student_probs + 1e-10))
        kl = kl.sum(dim=-1)  # [B, T-1, n_agents]
        distill_loss = (kl * mem_mask).sum() / mem_mask.sum().clamp(min=1.0)

        return distill_loss

    # ═══════════════════════════════════════════════════════════
    # Target Network Updates
    # ═══════════════════════════════════════════════════════════

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(
            self.target_mac.parameters(), self.mac.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        for target_param, param in zip(
            self.target_mixer.parameters(), self.mixer.parameters()
        ):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def _update_teacher_mac(self):
        """EMA update of teacher network for CL distillation."""
        for teacher_param, param in zip(
            self.cl_teacher_mac.agent.parameters(),
            self.mac.agent.parameters(),
        ):
            teacher_param.data.copy_(
                teacher_param.data * (1.0 - self.cl_teacher_ema_tau)
                + param.data * self.cl_teacher_ema_tau
            )

    # ═══════════════════════════════════════════════════════════
    # Utilities
    # ═══════════════════════════════════════════════════════════

    def _freeze_module(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad_(False)

    def _freeze_mac(self, mac):
        mac.agent.eval()
        for param in mac.agent.parameters():
            param.requires_grad_(False)

    # ═══════════════════════════════════════════════════════════
    # Device / Save / Load
    # ═══════════════════════════════════════════════════════════

    def cuda(self):
        device = getattr(self.args, "device", "cuda")
        self.mac.cuda()
        self.target_mac.cuda()
        self.mixer.to(device)
        self.target_mixer.to(device)
        if self.dynamics_predictor is not None:
            self.dynamics_predictor.to(device)
        if self.cl_teacher_mac is not None:
            self.cl_teacher_mac.cuda()
            self._freeze_mac(self.cl_teacher_mac)

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.main_optimizer.state_dict(), "{}/opt.th".format(path))
        if self.dynamics_predictor is not None:
            th.save(self.dynamics_predictor.state_dict(), "{}/lgdd_predictor.th".format(path))
        if self.lgdd_optimizer is not None:
            th.save(self.lgdd_optimizer.state_dict(), "{}/lgdd_opt.th".format(path))
        if self.cl_teacher_mac is not None:
            th.save(self.cl_teacher_mac.agent.state_dict(), "{}/cl_teacher.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        self.mixer.load_state_dict(
            th.load(
                "{}/mixer.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.main_optimizer.load_state_dict(
            th.load(
                "{}/opt.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )
        if self.dynamics_predictor is not None:
            predictor_path = "{}/lgdd_predictor.th".format(path)
            if os.path.exists(predictor_path):
                self.dynamics_predictor.load_state_dict(
                    th.load(predictor_path, map_location=lambda storage, loc: storage)
                )
        if self.lgdd_optimizer is not None:
            lgdd_opt_path = "{}/lgdd_opt.th".format(path)
            if os.path.exists(lgdd_opt_path):
                self.lgdd_optimizer.load_state_dict(
                    th.load(lgdd_opt_path, map_location=lambda storage, loc: storage)
                )
        if self.cl_teacher_mac is not None:
            teacher_path = "{}/cl_teacher.th".format(path)
            if os.path.exists(teacher_path):
                self.cl_teacher_mac.agent.load_state_dict(
                    th.load(teacher_path, map_location=lambda storage, loc: storage)
                )
            self._freeze_mac(self.cl_teacher_mac)
