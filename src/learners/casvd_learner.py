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

        # ── α annealing schedule ──
        # Diagnoses & fixes the late-training stalling regression: once the
        # policy's win rate saturates, the constant entropy bonus dominates
        # the optimisation signal and pulls the team into stalling.  We
        # decay α from `entropy_coef` toward `entropy_anneal_floor` over the
        # window [entropy_anneal_start, entropy_anneal_end].  Default off
        # (matches vanilla Soft-QMIX behaviour).
        self.entropy_anneal_enabled = getattr(args, "entropy_anneal_enabled", False)
        self.entropy_anneal_start   = int(getattr(args, "entropy_anneal_start", 6_000_000))
        self.entropy_anneal_end     = int(getattr(args, "entropy_anneal_end",   9_000_000))
        self.entropy_anneal_floor   = float(getattr(args, "entropy_anneal_floor", 0.005))
        assert self.entropy_anneal_end >= self.entropy_anneal_start >= 0, (
            f"entropy_anneal_end ({self.entropy_anneal_end}) must be ≥ "
            f"entropy_anneal_start ({self.entropy_anneal_start}) ≥ 0"
        )
        assert 0.0 <= self.entropy_anneal_floor <= self.entropy_coef, (
            f"entropy_anneal_floor ({self.entropy_anneal_floor}) must be in "
            f"[0, entropy_coef={self.entropy_coef}]"
        )

        # ── Per-agent adaptive alpha (off by default) ──
        self.use_adaptive_alpha = getattr(args, "use_adaptive_alpha", False)
        self._coord_signals = None
        self.coord_signal_ema_tau = getattr(args, "coord_signal_ema_tau", 0.99)

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

    def _current_alpha(self, t_env):
        """Annealed entropy coefficient at training step t_env.

        Schedule:
          t < anneal_start            → α = entropy_coef     (full bonus)
          t in [anneal_start, anneal_end] → linear decay to floor
          t ≥ anneal_end              → α = anneal_floor     (near-greedy)

        With annealing disabled, returns the original scalar entropy_coef
        so behaviour is identical to vanilla Soft-QMIX.
        """
        if not self.entropy_anneal_enabled:
            return self.entropy_coef
        if t_env <= self.entropy_anneal_start:
            return self.entropy_coef
        if t_env >= self.entropy_anneal_end:
            return self.entropy_anneal_floor
        span = max(1, self.entropy_anneal_end - self.entropy_anneal_start)
        frac = float(t_env - self.entropy_anneal_start) / float(span)
        return self.entropy_coef + frac * (self.entropy_anneal_floor - self.entropy_coef)

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
        for t in range(current_batch.max_seq_length):
            q_values = self.mac.forward(current_batch, t)
            mac_out.append(q_values)
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

            # ── Annealed α used in BOTH sampling and target ──
            # Computed once per train step so policy temperature, target
            # entropy bonus, and rollout selector all stay consistent.
            alpha_t = self._current_alpha(t_env)

            # Soft policy from online net (Double-Q: online selects)
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach = self.mixer.func_f(mac_out_detach, states, t_env)
            mac_out_detach = mac_out_detach / alpha_t
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

            # TD(λ) with entropy bonus — uses the same annealed α as sampling
            targets = build_td_lambda_targets(
                rewards, terminated, mask,
                target_qvals, self.n_agents,
                self.args.gamma, self.args.td_lambda,
                target_entropy=target_entropy * alpha_t,
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
        # 7. Backward + optimise
        # ═══════════════════════════════════════════════════════
        self.main_optimizer.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.main_params, self.args.grad_norm_clip)
        self.main_optimizer.step()

        # Push the current annealed α to the action selector so the next
        # rollout batch samples actions with the same temperature used in
        # this train step.  Required for the target/rollout consistency
        # that Double-Q assumes.  Cheap — sets a single Python float.
        if self.entropy_anneal_enabled:
            if hasattr(self.mac, "set_alpha"):
                self.mac.set_alpha(alpha_t)
            elif hasattr(self.mac.action_selector, "entropy_coef"):
                self.mac.action_selector.entropy_coef = alpha_t

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
            # Annealed α actually used this step (= entropy_coef when annealing
            # is off; decays toward `entropy_anneal_floor` when on).
            self.logger.log_stat("alpha_t", alpha_t, t_env)
            self.logger.log_stat(
                "err_mask", (mask_sum.sum() / mask_exp.sum()).item(), t_env
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
