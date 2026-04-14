import copy
import math
import os

import torch as th
import torch.nn.functional as F
from torch.optim import Adam

from modules.mixers.nmix import Mixer as NMixer
from utils.rl_utils import build_td_lambda_targets


class CASVDLearner:
    """Coordination-Aware Soft Value Decomposition learner.

    Combines:
    - GAT encoder with parameter sharing (shared Q-head, no tanh squashing)
    - QMIX mixer for value decomposition with monotonicity
    - Q-spread-relative soft value targets (no entropy bonus):
        α_i = max(alpha_factor_i × ΔQ_i, alpha_floor)
        V_soft(s,i) = E_{π_soft}[Q_target(s,i,a)]
      Self-stabilizing: small Q-spread → small α → V_soft ≈ V_hard.
      Scale-invariant: softmax shape constant regardless of Q-value scale.
    - Per-agent alpha_factor from coordination signal:
        alpha_factor_i = α_min + (α_max - α_min) × coord_signal_i
        coord_signal_i = 0 → well-coordinated → near hard-max (exploit)
        coord_signal_i = 1 → poorly-coordinated → softer targets (explore)
    - InfoNCE coordination sensor (gradient-isolated, uses GRU hidden states):
        Agent i predicts mean of OTHER agents' hidden states at t+1.
        Hidden states encode action-observation history → naturally distinct
        per agent → genuine coord_signal differentiation.
    - Continual learning with teacher distillation
    """

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions

        # Target network
        self.target_mac = copy.deepcopy(self.mac)

        # QMIX-style mixer from nmix.py
        self.mixer = NMixer(args)
        self.target_mixer = copy.deepcopy(self.mixer)

        # ── Option B: Soft Bellman backup via logsumexp ──
        # V_soft(s,i) = entropy_coef · log Σ_a exp(Q_target(s,i,a) / entropy_coef)
        #             = E_π[Q_target] + entropy_coef · H(π)
        # Proper entropy-regularised soft value (V_soft ≥ max Q by α·H(π)),
        # replacing the previous broken E_π[Q_target] form which was strictly
        # ≤ max Q and therefore strictly worse than hard-max QMIX.
        self.use_soft_values = getattr(args, "use_soft_values", False)
        self.entropy_coef = getattr(args, "entropy_coef", 0.1)

        # Legacy (Option A) scale-invariant params — retained only so that
        # loading an old checkpoint / config does not crash.  They are NOT
        # used by the soft-value branch below when use_adaptive_alpha=False.
        self.alpha_factor = getattr(args, "alpha_factor_init", 0.5)
        self.alpha_factor_min = getattr(args, "alpha_factor_min", 0.0)
        self.alpha_factor_max = getattr(args, "alpha_factor_max", 0.3)
        self.alpha_floor = getattr(args, "alpha_floor", 0.005)
        self.soft_value_warmup_steps = getattr(args, "soft_value_warmup_steps", 0)

        # ── Per-agent adaptive alpha_factor ──
        # Each agent gets its own alpha_factor from its InfoNCE coordination score.
        # alpha_factor_i = α_min + (α_max - α_min) × coord_signal_i
        self.use_adaptive_alpha = getattr(args, "use_adaptive_alpha", False)
        # Per-agent coord signals, stored as [n_agents] tensor (EMA-smoothed)
        self._coord_signals = None  # lazily initialized
        self.coord_signal_ema_tau = getattr(args, "coord_signal_ema_tau", 0.99)

        # ── InfoNCE coordination sensor (gradient-isolated) ──
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

        # ── Continual learning ──
        self.cl_enabled = getattr(args, "cl_enabled", False)
        self.cl_distill_weight = getattr(args, "cl_distill_weight", 0.0)
        self.cl_teacher_ema_tau = getattr(args, "cl_teacher_ema_tau", 0.002)
        self.cl_teacher_mac = None

        if self.cl_enabled and self.cl_distill_weight > 0:
            self.cl_teacher_mac = copy.deepcopy(self.mac)
            self._freeze_mac(self.cl_teacher_mac)

        # ── Optimizers: fully separate main and LGDD ──
        # Main optimizer: encoder + Q-head + mixer (TD + CL gradients)
        # LGDD optimizer: predictor MLP only (LGDD gradients, isolated)
        opt_eps = getattr(args, "optimizer_epsilon", 1e-7)
        self.main_params = list(self.mac.parameters()) + list(self.mixer.parameters())
        self.main_optimizer = Adam(self.main_params, lr=args.lr, eps=opt_eps)

        if self.dynamics_params:
            lgdd_lr = getattr(args, "lgdd_lr", args.lr)
            self.lgdd_optimizer = Adam(self.dynamics_params, lr=lgdd_lr, eps=opt_eps)
        else:
            self.lgdd_optimizer = None

        # For grad clipping compatibility
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

    def train(self, batch, t_env, episode_num, current_batch_size=None, memory_batch_size=0):
        # ── Split batch: TD+LGDD on current portion only ──
        actual_current = current_batch_size if current_batch_size is not None else batch.batch_size
        if memory_batch_size > 0 and batch.batch_size > actual_current:
            current_batch = batch[:actual_current]
            memory_batch = batch[actual_current:]
        else:
            current_batch = batch
            memory_batch = None

        rewards = current_batch["reward"][:, :-1]                        # [B, T-1, 1]
        actions = current_batch["actions"][:, :-1]                       # [B, T-1, n_agents, 1]
        terminated = current_batch["terminated"][:, :-1].float()         # [B, T-1, 1]
        mask = current_batch["filled"][:, :-1].float()                   # [B, T-1, 1]
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = current_batch["avail_actions"]                   # [B, T, n_agents, n_actions]

        # ═══════════════════════════════════════════════════════
        # 1. Forward pass: Q-values + latents for LGDD
        # ═══════════════════════════════════════════════════════
        self.mac.init_hidden(current_batch.batch_size)
        mac_out = []
        all_latents = []
        all_hidden = []   # GRU hidden states for InfoNCE (more agent-specific than local_summary)

        for t in range(current_batch.max_seq_length):
            if self.lgdd_enabled:
                q_values, latents = self.mac.forward_with_latents(current_batch, t)
                all_latents.append(latents)
                all_hidden.append(self.mac.hidden_states.detach().clone())
            else:
                q_values = self.mac.forward(current_batch, t)
            mac_out.append(q_values)

        mac_out = th.stack(mac_out, dim=1)  # [B, T, n_agents, n_actions]

        # Chosen action Q-values
        chosen_action_qvals = th.gather(
            mac_out[:, :-1], dim=3, index=actions
        ).squeeze(3)  # [B, T-1, n_agents]

        # ═══════════════════════════════════════════════════════
        # 2. Target Q-values with double-Q and td_lambda
        # ═══════════════════════════════════════════════════════
        alpha_mean_for_log   = 0.0
        q_spread_mean_for_log = 0.0
        entropy_mean_for_log  = 0.0

        with th.no_grad():
            self.target_mac.init_hidden(current_batch.batch_size)
            target_mac_out = []
            for t in range(current_batch.max_seq_length):
                target_q = self.target_mac.forward(current_batch, t)
                target_mac_out.append(target_q)
            target_mac_out = th.stack(target_mac_out, dim=1)  # [B, T, n_agents, n_actions]

            if self.use_soft_values:
                # ── Option B: Soft Bellman backup via logsumexp ───────────────
                #
                # V_soft(s,i) = α · log Σ_a exp(Q_target(s,i,a) / α)
                #             = E_{π_soft}[Q_target(s,i,a)] + α · H(π_soft)
                #
                # This is the proper entropy-regularised soft value.  Unlike
                # the previous E_π[Q_target] form (always ≤ max Q, strictly
                # worse than hard-max), this satisfies V_soft ≥ max_a Q_target
                # by α · H(π).  Known contraction to the soft-optimal value
                # (standard SAC / soft-Q-learning backup).
                #
                # α is fixed at self.entropy_coef.  The q-spread-relative
                # adaptive machinery has been removed — it was scale-invariant
                # (not exploration-when-uncertain) and pinned at the floor.
                alpha = self.entropy_coef

                # Mask unavailable actions before logsumexp.  -1e10 / α is
                # still a finite number (e.g. -1e11 at α=0.1); log_softmax
                # and logsumexp stay numerically stable because exp(-1e11)
                # underflows cleanly to 0 without producing NaNs.
                target_q_masked = target_mac_out.clone()
                target_q_masked[avail_actions == 0] = -1e10

                # Per-agent soft value
                v_soft = alpha * th.logsumexp(
                    target_q_masked / alpha, dim=-1
                )
                # v_soft: [B, T, n_agents]

                # ── Diagnostics only (not part of the loss) ────────────────
                # Log the online soft policy's entropy and the fixed α so we
                # can monitor whether the soft-value branch is actually doing
                # anything (compared to hard-max).
                online_q = mac_out.clone().detach()
                online_q[avail_actions == 0] = -1e10
                online_log_probs = th.log_softmax(online_q / alpha, dim=-1)
                online_probs = th.exp(online_log_probs)
                avail_f = avail_actions.float()
                entropy = -(online_probs * online_log_probs * avail_f).sum(dim=-1)
                # entropy: [B, T, n_agents]

                alpha_mean_for_log    = float(alpha)
                q_spread_mean_for_log = 0.0                  # unused in Option B
                entropy_mean_for_log  = entropy.mean().item()

                # Mix per-agent soft values through the target mixer
                target_q_total = self.target_mixer(
                    v_soft, current_batch["state"]
                )  # [B, T, 1]

                # TD(λ) multi-step returns
                targets = build_td_lambda_targets(
                    rewards, terminated, mask,
                    target_q_total, self.n_agents,
                    self.args.gamma, self.args.td_lambda,
                )  # [B, T-1, 1]

            else:
                # ── Standard hard max target (QMIX-style) ──
                mac_out_detach = mac_out.clone().detach()
                mac_out_detach[avail_actions == 0] = -1e10
                cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]

                target_max_qvals = th.gather(
                    target_mac_out, dim=3, index=cur_max_actions
                ).squeeze(3)  # [B, T, n_agents]

                target_max_qvals = self.target_mixer(
                    target_max_qvals, current_batch["state"]
                )  # [B, T, 1]

                targets = build_td_lambda_targets(
                    rewards, terminated, mask,
                    target_max_qvals, self.n_agents,
                    self.args.gamma, self.args.td_lambda,
                )  # [B, T-1, 1]

        # ═══════════════════════════════════════════════════════
        # 3. Mix chosen Q-values through QMIX
        # ═══════════════════════════════════════════════════════
        agent_chosen_qvals = chosen_action_qvals
        chosen_action_qvals = self.mixer(
            chosen_action_qvals, current_batch["state"][:, :-1]
        )  # [B, T-1, 1]

        # ═══════════════════════════════════════════════════════
        # 4. TD loss
        # ═══════════════════════════════════════════════════════
        td_error = chosen_action_qvals - targets.detach()
        masked_td_error = 0.5 * (td_error ** 2) * mask
        td_loss = masked_td_error.sum() / mask.sum()

        # ═══════════════════════════════════════════════════════
        # 5. InfoNCE coordination sensor (gradient-isolated)
        #    Measures mutual information between each agent's local
        #    embedding and the team's global future state.
        #    Encoder outputs are DETACHED — InfoNCE gradients never
        #    reach the encoder. Only the bilinear W matrix is trained.
        # ═══════════════════════════════════════════════════════
        infonce_loss = th.tensor(0.0, device=current_batch.device)
        infonce_error = 0.0
        coord_signal_mean = 0.0

        # Embedding stat holders — populated inside lgdd block, logged below
        local_emb_mean        = 0.0
        local_emb_std         = 0.0
        local_emb_min         = 0.0
        local_emb_max         = 0.0
        local_emb_norm_mean   = 0.0
        team_emb_mean         = 0.0
        team_emb_std          = 0.0
        team_emb_min          = 0.0
        team_emb_max          = 0.0
        team_emb_norm_mean    = 0.0
        global_emb_mean       = 0.0
        global_emb_std        = 0.0
        global_emb_min        = 0.0
        global_emb_max        = 0.0
        global_emb_norm_mean  = 0.0
        local_emb_dead_frac   = 0.0   # fraction of near-zero local embeddings (collapsed dims)

        if self.lgdd_enabled:
            T = current_batch.max_seq_length
            mask_squeezed = mask.squeeze(-1)   # [B, T-1]

            # ── GRU hidden states for InfoNCE ─────────────────────────
            # Hidden states encode action-observation history → naturally
            # distinct per agent (unlike local_summary which is a snapshot
            # of current observations and tends to be similar across agents
            # in the same battle). This enables genuine per-agent coord_signal
            # differentiation.
            all_h = th.stack(all_hidden, dim=0)  # [T, B, n_agents, D]

            # ── Local GAT embedding stats (kept for diagnostics) ──────
            all_local = th.stack(
                [lat["local_summary"].detach() for lat in all_latents], dim=0
            )  # [T, B, n_agents, D]
            local_emb_mean       = all_local.mean().item()
            local_emb_std        = all_local.std().item()
            local_emb_min        = all_local.min().item()
            local_emb_max        = all_local.max().item()
            local_norms          = all_local.norm(dim=-1)
            local_emb_norm_mean  = local_norms.mean().item()
            local_emb_dead_frac  = (local_norms < 0.01).float().mean().item()

            # ── Team GAT embedding stats ───────────────────────────────
            if all_latents and "team_summary" in all_latents[0]:
                all_team = th.stack(
                    [lat["team_summary"].detach() for lat in all_latents], dim=0
                )  # [T, B, n_agents, D]
                team_emb_mean      = all_team.mean().item()
                team_emb_std       = all_team.std().item()
                team_emb_min       = all_team.min().item()
                team_emb_max       = all_team.max().item()
                team_emb_norm_mean = all_team.norm(dim=-1).mean().item()

            # ── InfoNCE targets from hidden states ─────────────────────
            # Global team mean at each timestep (for negatives)
            g_all = all_h.mean(dim=2)  # [T, B, D]

            # Per-agent "others mean": mean of OTHER agents' hidden states.
            # Agent i's target excludes itself → genuinely different across agents.
            if self.n_agents > 1:
                team_sum = all_h.sum(dim=2, keepdim=True)   # [T, B, 1, D]
                g_others = (team_sum - all_h) / (self.n_agents - 1)  # [T, B, n_agents, D]
            else:
                g_others = all_h  # degenerate: single agent

            # ── Global embedding stats (from hidden states now) ────────
            global_emb_mean      = g_all.mean().item()
            global_emb_std       = g_all.std().item()
            global_emb_min       = g_all.min().item()
            global_emb_max       = g_all.max().item()
            global_emb_norm_mean = g_all.norm(dim=-1).mean().item()

            # Compute InfoNCE loss for timesteps 0..T-2 (predicting t+1)
            K = self.infonce_n_negatives
            all_per_agent_loss = []

            for t in range(T - 1):
                h_i = all_h[t]               # [B, n_agents, D] — agent i's hidden state at t
                g_pos = g_others[t + 1]      # [B, n_agents, D] — per-agent others-future

                # Sample K negatives: global team mean at random timesteps.
                neg_indices = []
                for _ in range(K):
                    idx = th.randint(0, T, (1,)).item()
                    while idx == t + 1:
                        idx = th.randint(0, T, (1,)).item()
                    neg_indices.append(idx)
                g_neg = th.stack([g_all[i] for i in neg_indices], dim=1)  # [B, K, D]

                # Per-agent InfoNCE loss: [B, n_agents]
                loss_t = self.dynamics_predictor(h_i, g_pos, g_neg)
                all_per_agent_loss.append(loss_t)

            # Stack: [T-1, B, n_agents] → [B, T-1, n_agents]
            all_per_agent_loss = th.stack(all_per_agent_loss, dim=0).permute(1, 0, 2)

            # Masked average across time: [B, n_agents]
            mask_expanded = mask_squeezed.unsqueeze(-1)  # [B, T-1, 1]
            denom = mask_expanded.sum(dim=1).clamp(min=1.0)  # [B, 1]
            per_agent_loss = (all_per_agent_loss * mask_expanded).sum(dim=1) / denom  # [B, n_agents]

            # Overall InfoNCE loss for backward (mean across batch and agents)
            infonce_loss = per_agent_loss.mean()
            infonce_error = infonce_loss.item()

            # Per-agent coordination signal: normalized to [0, 1]
            max_infonce = math.log(K + 1)
            coord_signal_batch = (per_agent_loss.mean(dim=0) / max_infonce).clamp(0.0, 1.0)
            # coord_signal_batch: [n_agents] — per-agent, 0=coordinated, 1=random

            # EMA smooth per-agent signals
            tau = self.coord_signal_ema_tau
            cs = self._get_coord_signals(current_batch.device)
            self._coord_signals = tau * cs + (1.0 - tau) * coord_signal_batch.detach()
            coord_signal_mean = self._coord_signals.mean().item()

            # Also update scalar alpha_factor for logging
            self.alpha_factor = (
                self.alpha_factor_min
                + (self.alpha_factor_max - self.alpha_factor_min) * coord_signal_mean
            )

        # ═══════════════════════════════════════════════════════
        # 7. CL distillation on memory portion ONLY
        # ═══════════════════════════════════════════════════════
        cl_loss = th.tensor(0.0, device=current_batch.device)
        if self.cl_enabled and self.cl_distill_weight > 0 and memory_batch is not None:
            cl_loss = self._compute_cl_distillation(memory_batch)
            td_loss = td_loss + self.cl_distill_weight * cl_loss

        # ═══════════════════════════════════════════════════════
        # 8. Two separate backward passes (gradient isolation)
        #    Pass 1: LGDD predictor only (detached inputs → no encoder grads)
        #    Pass 2: Encoder + mixer + Q-head (TD + CL)
        # ═══════════════════════════════════════════════════════

        # Pass 1: InfoNCE predictor update (independent computation graph)
        if self.lgdd_enabled and self.lgdd_optimizer is not None:
            self.lgdd_optimizer.zero_grad()
            infonce_loss.backward()
            th.nn.utils.clip_grad_norm_(self.dynamics_params, self.args.grad_norm_clip)
            self.lgdd_optimizer.step()

        # Pass 2: Main network update (encoder + mixer)
        self.main_optimizer.zero_grad()
        td_loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.main_params, self.args.grad_norm_clip)
        self.main_optimizer.step()

        # ═══════════════════════════════════════════════════════
        # 9. Target network updates
        # ═══════════════════════════════════════════════════════
        tau = self.args.target_update_interval_or_tau
        if tau > 1:
            if (episode_num - self.last_target_update_episode) / tau >= 1.0:
                self._update_targets_hard()
                self.last_target_update_episode = episode_num
        else:
            self._update_targets_soft(tau)

        # Teacher MAC EMA update (for CL distillation)
        if self.cl_teacher_mac is not None:
            self._update_teacher_mac()

        # ═══════════════════════════════════════════════════════
        # 10. Logging
        # ═══════════════════════════════════════════════════════
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            agent_q_taken_mean = (agent_chosen_qvals * mask).sum().item() / (mask.sum().item() * self.n_agents)
            agent_q_mean = mac_out.mean().item()
            agent_q_std = mac_out.std().item()
            agent_q_max_abs = mac_out.abs().max().item()
            self.logger.log_stat("loss", td_loss.item(), t_env)
            self.logger.log_stat(
                "grad_norm",
                grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
                t_env,
            )
            self.logger.log_stat("agent_q_taken_mean", agent_q_taken_mean, t_env)
            self.logger.log_stat("agent_q_mean", agent_q_mean, t_env)
            self.logger.log_stat("agent_q_std", agent_q_std, t_env)
            self.logger.log_stat("agent_q_max_abs", agent_q_max_abs, t_env)
            self.logger.log_stat(
                "q_taken_mean",
                (chosen_action_qvals * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            self.logger.log_stat(
                "target_mean",
                (targets * mask).sum().item() / mask.sum().item(),
                t_env,
            )
            if self.use_soft_values:
                # Option B: fixed α (entropy_coef), no q-spread, no adaptive factor.
                # `alpha_factor` / `q_spread_mean` are legacy log keys kept at
                # zero so downstream log parsers don't break.
                self.logger.log_stat("entropy_coef", float(self.entropy_coef), t_env)
                self.logger.log_stat("alpha_mean", alpha_mean_for_log, t_env)
                self.logger.log_stat("alpha_factor", 0.0, t_env)
                self.logger.log_stat("q_spread_mean", q_spread_mean_for_log, t_env)
                self.logger.log_stat("entropy_mean", entropy_mean_for_log, t_env)
            if self.lgdd_enabled:
                self.logger.log_stat("infonce_loss", infonce_loss.item(), t_env)
                self.logger.log_stat("coord_signal_mean", coord_signal_mean, t_env)
                if self._coord_signals is not None:
                    self.logger.log_stat("coord_signal_std", self._coord_signals.std().item(), t_env)

                # ── Local GAT embedding stats ──────────────────────────
                self.logger.log_stat("local_emb_mean",      local_emb_mean,      t_env)
                self.logger.log_stat("local_emb_std",       local_emb_std,       t_env)
                self.logger.log_stat("local_emb_min",       local_emb_min,       t_env)
                self.logger.log_stat("local_emb_max",       local_emb_max,       t_env)
                self.logger.log_stat("local_emb_norm_mean", local_emb_norm_mean, t_env)
                self.logger.log_stat("local_emb_dead_frac", local_emb_dead_frac, t_env)

                # ── Team GAT embedding stats ───────────────────────────
                self.logger.log_stat("team_emb_mean",       team_emb_mean,       t_env)
                self.logger.log_stat("team_emb_std",        team_emb_std,        t_env)
                self.logger.log_stat("team_emb_min",        team_emb_min,        t_env)
                self.logger.log_stat("team_emb_max",        team_emb_max,        t_env)
                self.logger.log_stat("team_emb_norm_mean",  team_emb_norm_mean,  t_env)

                # ── Global team embedding stats (mean-pooled local) ────
                self.logger.log_stat("global_emb_mean",     global_emb_mean,     t_env)
                self.logger.log_stat("global_emb_std",      global_emb_std,      t_env)
                self.logger.log_stat("global_emb_min",      global_emb_min,      t_env)
                self.logger.log_stat("global_emb_max",      global_emb_max,      t_env)
                self.logger.log_stat("global_emb_norm_mean",global_emb_norm_mean,t_env)
            if self.cl_enabled and self.cl_distill_weight > 0:
                self.logger.log_stat("cl_distill_loss", cl_loss.item(), t_env)
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
