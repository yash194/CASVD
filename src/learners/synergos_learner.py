"""SynergosLearner — full SYNERGOS algorithm (Phases 1 + 2 + 3 + 4 + 5).

Each phase is independently toggleable via YAML so the file doubles as the
implementation of every ablation in the paper.

────────────────────────────────────────────────────────────────────────────
   Phase  Toggle               Component                                Δ-pp
────────────────────────────────────────────────────────────────────────────
   1     (always on)          QR-DQN distributional Z + CVaR-Soft     +3-6
                              policy + quantile-Huber TD loss
   2     use_sheaf            Sheaf-cochain mixing across agents      +2-4
   3     use_dual_alpha       Per-agent Lagrangian α (ADER-style)     +1-2
   4     use_synergy_bonus    PID-S intrinsic reward (NOVEL)          +1-3
   5a    use_slow_role        Future-conditioned slow-role InfoNCE    +1-2
   5b    use_sync_loss        Multi-timescale predictive synchrony    +0.5-2
────────────────────────────────────────────────────────────────────────────

Architectural invariants.
=========================
  • Single online + target network; standard CTDE.
  • Main RL loss (quantile-Huber + β-loss) and each auxiliary loss
    (synergy, slow-role, sync) trained with their *own* optimisers,
    `.detach()`-isolated so RL gradients never flow into auxiliary
    discriminators and vice versa.
  • IGM holds at every quantile via softplus restriction maps + convex
    sheaf updates + sum aggregation (proposal §5.1).
  • Synergy bonus is potential-shaped: η · max S̃ < (1 − γ) · α · log|A|
    is enforced by the η · clip combination in the YAML defaults.

Optimisers (per phase).
========================
  • main_optimizer        : MAC + mixer    (TD + β + sheaf params)
  • alpha_optimizer       : log_α [N]      (Phase 3)
  • synergy_optimizer     : synergy NCE    (Phase 4)
  • slow_role_optimizer   : slow-role NCE  (Phase 5a)
  • sync_optimizer        : sync MLPs      (Phase 5b)
"""
import copy
import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from modules.mixers.dist_soft_mix import DistSoftMixer
from modules.mixers.sheaf_soft_mix import SheafSoftMixer
from utils.rl_utils import build_td_lambda_targets


class SynergosLearner:
    # ═══════════════════════════════════════════════════════════════════
    # Init
    # ═══════════════════════════════════════════════════════════════════

    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.K = int(getattr(args, "n_quantiles", 8))

        # ── Phase toggles ────────────────────────────────────────
        self.use_sheaf          = bool(getattr(args, "use_sheaf",          True))
        self.use_dual_alpha     = bool(getattr(args, "use_dual_alpha",     True))
        self.use_synergy_bonus  = bool(getattr(args, "use_synergy_bonus",  True))
        self.use_slow_role      = bool(getattr(args, "use_slow_role",      True))
        self.use_sync_loss      = bool(getattr(args, "use_sync_loss",      True))

        # Hidden-state capture is required for any Phase-5 component.
        self.capture_hidden = self.use_slow_role or self.use_sync_loss

        # ── Phase 1: distributional + Soft-QMIX backbone ─────────
        self.taus = th.tensor(
            [(2 * k - 1) / (2.0 * self.K) for k in range(1, self.K + 1)],
            dtype=th.float32,
        )
        self.entropy_coef = float(getattr(args, "entropy_coef", 0.03))
        self.huber_kappa  = float(getattr(args, "huber_kappa", 1.0))

        self.cvar_beta_start  = float(getattr(args, "cvar_beta_start", 1.0))
        self.cvar_beta_end    = float(getattr(args, "cvar_beta_end",   0.25))
        self.cvar_anneal_start = int(getattr(args, "cvar_anneal_start", 0))
        self.cvar_anneal_end   = int(getattr(args, "cvar_anneal_end",   1_000_000))

        # Mixer choice — Phase 2 toggle.
        if self.use_sheaf:
            self.mixer = SheafSoftMixer(args)
        else:
            self.mixer = DistSoftMixer(args)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.target_mac   = copy.deepcopy(self.mac)
        self.mac.set_mixer(self.mixer)

        # ── Phase 3: per-agent dual α (Lagrangian) ───────────────
        if self.use_dual_alpha:
            log_alpha_init = math.log(self.entropy_coef)
            self.log_alpha = nn.Parameter(
                th.full((self.n_agents,), log_alpha_init, dtype=th.float32),
                requires_grad=True,
            )
            target_ratio = float(getattr(args, "target_entropy_ratio", 0.5))
            # Per-agent target entropy.  log|A| is the entropy of a uniform
            # policy; ratio < 1 says "we want the policy to commit somewhat".
            self.target_entropy = target_ratio * math.log(self.n_actions)
            alpha_lr = float(getattr(args, "alpha_lr", 3e-4))
            self.alpha_optimizer = Adam([self.log_alpha], lr=alpha_lr)
            # α floor / ceiling guards — keeps log_α from drifting to extremes.
            self.alpha_min = float(getattr(args, "alpha_min", 1e-3))
            self.alpha_max = float(getattr(args, "alpha_max", 1.0))

        # ── Phase 4: synergy estimator (PID-S) ───────────────────
        if self.use_synergy_bonus:
            from modules.predictors import SynergyEstimator
            state_dim = int(self.mixer.state_dim)
            self.synergy_estimator = SynergyEstimator(
                state_dim=state_dim,
                n_actions=self.n_actions,
                n_agents=self.n_agents,
                hidden=int(getattr(args, "synergy_hidden", 64)),
                temperature=float(getattr(args, "synergy_temperature", 0.5)),
            )
            self.synergy_eta   = float(getattr(args, "synergy_eta", 0.001))
            self.synergy_clip  = float(getattr(args, "synergy_clip", 0.5))
            self.synergy_n_neg = int(getattr(args, "synergy_n_negatives", 15))
            synergy_lr = float(getattr(args, "synergy_lr", 3e-4))
            self.synergy_optimizer = Adam(
                self.synergy_estimator.parameters(), lr=synergy_lr,
            )

        # ── Phase 5a: slow-role InfoNCE ──────────────────────────
        if self.use_slow_role:
            from modules.predictors import SlowRolePredictor
            self.slow_role = SlowRolePredictor(
                hidden_dim=int(getattr(args, "hidden_dim", 128)),
                role_dim=int(getattr(args, "slow_role_dim", 32)),
                n_agents=self.n_agents,
                temperature=float(getattr(args, "slow_role_temperature", 0.2)),
            )
            self.slow_role_horizon = int(getattr(args, "slow_role_horizon", 8))
            self.lambda_slow_role  = float(getattr(args, "lambda_slow_role",  0.1))
            slow_role_lr = float(getattr(args, "slow_role_lr", 3e-4))
            self.slow_role_optimizer = Adam(
                self.slow_role.parameters(), lr=slow_role_lr,
            )

        # ── Phase 5b: multi-timescale sync ───────────────────────
        if self.use_sync_loss:
            from modules.predictors import SyncPredictor
            self.sync_predictor = SyncPredictor(
                hidden_dim=int(getattr(args, "hidden_dim", 128)),
                n_bands=2,
                n_agents=self.n_agents,
            )
            self.sync_tau_fast = float(getattr(args, "sync_tau_fast", 0.95))
            self.sync_tau_slow = float(getattr(args, "sync_tau_slow", 0.99))
            self.lambda_sync   = float(getattr(args, "lambda_sync", 0.01))
            sync_lr = float(getattr(args, "sync_lr", 3e-4))
            self.sync_optimizer = Adam(
                self.sync_predictor.parameters(), lr=sync_lr,
            )

        # ── Main optimiser (MAC + mixer) ─────────────────────────
        opt_eps = getattr(args, "optimizer_epsilon", 1e-7)
        self.params = list(self.mac.parameters()) + list(self.mixer.parameters())
        self.main_optimizer = Adam(self.params, lr=args.lr, eps=opt_eps)

        self.last_target_update_episode = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        # Push initial α to the action selector.
        self._push_alpha_to_selector()

    # ═══════════════════════════════════════════════════════════════════
    # α handling (Phase 3 + scalar fallback)
    # ═══════════════════════════════════════════════════════════════════

    def _current_alpha_vec(self, device):
        """Return the per-agent α tensor [N] on `device` for sampling and
        target-entropy computation.  In Phase-3 mode this is exp(log_α)
        clamped to [alpha_min, alpha_max]; in scalar mode it is a uniform
        broadcast of self.entropy_coef."""
        if self.use_dual_alpha:
            alpha = self.log_alpha.exp().clamp(self.alpha_min, self.alpha_max)
            return alpha.to(device)
        return th.full(
            (self.n_agents,), float(self.entropy_coef), device=device,
        )

    def _push_alpha_to_selector(self):
        if not hasattr(self.mac, "set_alpha"):
            return
        if self.use_dual_alpha:
            with th.no_grad():
                self.mac.set_alpha(self._current_alpha_vec(
                    next(self.mac.parameters()).device
                ).detach().clone())
        else:
            self.mac.set_alpha(self.entropy_coef)

    # ═══════════════════════════════════════════════════════════════════
    # CVaR / loss helpers
    # ═══════════════════════════════════════════════════════════════════

    def _current_cvar_beta(self, t_env):
        if t_env <= self.cvar_anneal_start:
            return self.cvar_beta_start
        if t_env >= self.cvar_anneal_end:
            return self.cvar_beta_end
        span = max(1, self.cvar_anneal_end - self.cvar_anneal_start)
        frac = float(t_env - self.cvar_anneal_start) / float(span)
        return self.cvar_beta_start + frac * (self.cvar_beta_end - self.cvar_beta_start)

    @staticmethod
    def _cvar(z, beta):
        K = z.shape[-1]
        m = max(1, int(round(float(beta) * K)))
        z_sorted, _ = th.sort(z, dim=-1)
        return z_sorted[..., :m].mean(dim=-1)

    def _quantile_huber_loss(self, pred, target, mask):
        """Quantile Huber loss with QR-DQN's fixed τ.

        pred:   [B, T-1, K]   predicted quantile values  (Z_tot at chosen action)
        target: [B, T-1, K]   target  quantile values     (TD(λ) per quantile)
        mask:   [B, T-1, 1]
        """
        K = pred.shape[-1]
        kappa = self.huber_kappa
        pred_exp   = pred.unsqueeze(-1)
        target_exp = target.unsqueeze(-2)
        delta = target_exp - pred_exp
        abs_delta = delta.abs()
        huber = th.where(
            abs_delta <= kappa,
            0.5 * delta.pow(2),
            kappa * (abs_delta - 0.5 * kappa),
        )
        if self.taus.device != delta.device:
            self.taus = self.taus.to(delta.device)
        taus = self.taus.view(1, 1, K, 1)
        indicator = (delta < 0.0).float()
        rho = (taus - indicator).abs() * huber / kappa
        loss_per_step = rho.sum(dim=-1).mean(dim=-1, keepdim=True)
        return (loss_per_step * mask).sum() / mask.sum().clamp(min=1.0)

    # ═══════════════════════════════════════════════════════════════════
    # Mixer dispatch (Phase 2 toggle)
    # ═══════════════════════════════════════════════════════════════════

    def _mix_distributional(self, z_per_agent, states, target=False):
        """Mix per-agent distributional Q to Z_tot per quantile.

        z_per_agent: [B, T, N, K]  (per-agent at chosen action)
        states:      [B, T, S]
        """
        if self.use_sheaf:
            mixer = self.target_mixer if target else self.mixer
            return mixer.forward_dist_sheaf(z_per_agent, states)
        # Plain VDN sum (non-sheaf path).
        return z_per_agent.sum(dim=2)

    def _mix_scalar(self, q_per_agent, states, target=False):
        """Mix per-agent scalar Q (used only by the β-loss path)."""
        if self.use_sheaf:
            mixer = self.target_mixer if target else self.mixer
            return mixer.forward_sheaf(q_per_agent, states)
        return q_per_agent.sum(dim=-1, keepdim=True)

    # ═══════════════════════════════════════════════════════════════════
    # Main train step
    # ═══════════════════════════════════════════════════════════════════

    def train(self, batch, t_env, episode_num, **_kwargs):
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        states = batch["state"]

        bs = batch.batch_size
        T = batch.max_seq_length
        N = self.n_agents
        K = self.K

        # ═══════════════════════════════════════════════════════
        # 1. Online forward — distributional Z + (optional) latents
        # ═══════════════════════════════════════════════════════
        self.mac.init_hidden(bs)
        mac_out = []
        hidden_history = [] if self.capture_hidden else None
        for t in range(T):
            z_t = self.mac.forward(batch, t)                               # [B, N, A, K]
            mac_out.append(z_t)
            if self.capture_hidden:
                hidden_history.append(self.mac.hidden_states.detach().clone())
        mac_out = th.stack(mac_out, dim=1)                                 # [B, T, N, A, K]
        if self.capture_hidden:
            hidden_history = th.stack(hidden_history, dim=1)               # [B, T, N, D]

        mac_out = self.mixer.func_g_dist(mac_out, states, t_env)

        # Chosen-action distribution
        actions_exp = actions.unsqueeze(-1).expand(-1, -1, -1, -1, K)
        chosen_Z = th.gather(mac_out[:, :-1], dim=3, index=actions_exp).squeeze(3)
        # chosen_Z: [B, T-1, N, K]

        # ═══════════════════════════════════════════════════════
        # 2. Target — distributional Z under target net + soft policy
        # ═══════════════════════════════════════════════════════
        with th.no_grad():
            self.target_mac.init_hidden(bs)
            target_mac_out = []
            for t in range(T):
                z_t = self.target_mac.forward(batch, t)
                target_mac_out.append(z_t)
            target_mac_out = th.stack(target_mac_out, dim=1)
            target_mac_out = self.target_mixer.func_g_dist(target_mac_out, states, t_env)

            # Soft policy from ONLINE net (Double-Q): apply func_f, take CVaR_β
            mac_out_for_policy = mac_out.clone().detach()
            mac_out_for_policy = self.mixer.func_f_dist(mac_out_for_policy, states, t_env)
            beta = self._current_cvar_beta(t_env)
            v_beta = self._cvar(mac_out_for_policy, beta)                  # [B, T, N, A]
            v_beta = v_beta.masked_fill(avail_actions == 0, -1e9)

            alpha_vec = self._current_alpha_vec(v_beta.device)             # [N]
            alpha_bcast = alpha_vec.view(1, 1, -1, 1)
            logits = v_beta / alpha_bcast
            actions_pdf = th.softmax(logits, dim=-1)                       # [B, T, N, A]

            rand_idx = th.rand(actions_pdf[..., :1].shape, device=actions_pdf.device)
            actions_cdf = th.cumsum(actions_pdf, dim=-1)
            rand_idx = th.clamp(rand_idx, 1e-6, 1 - 1e-6)
            picked_actions = th.searchsorted(actions_cdf, rand_idx)        # [B, T, N, 1]

            picked_exp = picked_actions.unsqueeze(-1).expand(-1, -1, -1, -1, K)
            target_Z = th.gather(target_mac_out, dim=3, index=picked_exp).squeeze(3)
            # [B, T, N, K]

            # Entropy bonus per agent — uses the SAMPLING α_vec (heterogeneous)
            # for the per-agent log π contribution.  Component-1 fix from
            # CASVD: target entropy uses scalar α_mean to avoid the late
            # stalling attractor when α is heterogeneous.
            target_logp = th.log(actions_pdf + 1e-10)
            target_logp = th.gather(target_logp, dim=3, index=picked_actions).squeeze(3)
            # [B, T, N]
            alpha_target_scalar = float(alpha_vec.mean().item())
            target_entropy = -alpha_target_scalar * target_logp.sum(-1, keepdim=True)
            # [B, T, 1]

            # Mix target Z (Phase 2: sheaf or plain VDN sum)
            target_Z_tot = self._mix_distributional(target_Z, states, target=True)
            # [B, T, K]

            targets_per_q = build_td_lambda_targets(
                rewards, terminated, mask,
                target_Z_tot, self.n_agents,
                self.args.gamma, self.args.td_lambda,
                target_entropy=target_entropy,
            )                                                              # [B, T-1, K]

        # ═══════════════════════════════════════════════════════
        # 3. (Phase 4)  Synergy bonus  →  reward shaping
        # ═══════════════════════════════════════════════════════
        synergy_loss_for_log = 0.0
        synergy_bonus_mean   = 0.0
        if self.use_synergy_bonus:
            synergy_loss, synergy_bonus_mean, targets_per_q = self._apply_synergy_bonus(
                states, actions, mask, targets_per_q, rewards, terminated, target_entropy,
                target_Z_tot,
            )
            synergy_loss_for_log = float(synergy_loss.detach().item())

        # ═══════════════════════════════════════════════════════
        # 4. Mix chosen Z (Phase 2: sheaf or VDN sum)
        # ═══════════════════════════════════════════════════════
        chosen_Z_mixed = self._mix_distributional(chosen_Z, states[:, :-1], target=False)
        # [B, T-1, K]

        # ═══════════════════════════════════════════════════════
        # 5. Quantile Huber loss + β loss
        # ═══════════════════════════════════════════════════════
        L_td = self._quantile_huber_loss(chosen_Z_mixed, targets_per_q, mask)

        # β-loss: keep func_f ≈ identity, on the per-quantile MEAN.
        chosen_Z_mean_per_agent = chosen_Z.mean(dim=-1)                    # [B, T-1, N]
        chosen_Z_mixed_mean = chosen_Z_mixed.mean(dim=-1, keepdim=True)
        affine_aq = self.mixer.func_f(
            chosen_Z_mean_per_agent.detach(), states[:, :-1], t_env,
        )
        approx_error = chosen_Z_mixed_mean.detach() - affine_aq.sum(-1, keepdim=True)
        L_beta = (0.5 * approx_error.pow(2) * mask).sum() / mask.sum().clamp(min=1.0)

        loss_main = L_td + L_beta

        # ═══════════════════════════════════════════════════════
        # 6. (Phase 5a) Slow-role InfoNCE
        # ═══════════════════════════════════════════════════════
        slow_role_loss_for_log = 0.0
        slow_role_top1 = 0.0
        if self.use_slow_role and hidden_history is not None:
            sr_loss, slow_role_top1 = self._compute_slow_role_loss(
                hidden_history, mask,
            )
            slow_role_loss_for_log = float(sr_loss.detach().item())
            self.slow_role_optimizer.zero_grad()
            sr_loss.backward()
            th.nn.utils.clip_grad_norm_(
                self.slow_role.parameters(), self.args.grad_norm_clip,
            )
            self.slow_role_optimizer.step()

        # ═══════════════════════════════════════════════════════
        # 7. (Phase 5b) Multi-timescale predictive sync
        # ═══════════════════════════════════════════════════════
        sync_loss_for_log = 0.0
        if self.use_sync_loss and hidden_history is not None:
            sync_loss = self._compute_sync_loss(hidden_history)
            sync_loss_for_log = float(sync_loss.detach().item())
            self.sync_optimizer.zero_grad()
            sync_loss.backward()
            th.nn.utils.clip_grad_norm_(
                self.sync_predictor.parameters(), self.args.grad_norm_clip,
            )
            self.sync_optimizer.step()

        # ═══════════════════════════════════════════════════════
        # 8. Main optimisation step
        # ═══════════════════════════════════════════════════════
        self.main_optimizer.zero_grad()
        loss_main.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.main_optimizer.step()

        # ═══════════════════════════════════════════════════════
        # 9. (Phase 3) Per-agent dual α — update from current policy entropy
        # ═══════════════════════════════════════════════════════
        if self.use_dual_alpha:
            # Recompute per-agent entropy from `actions_pdf` (computed during
            # target phase under no_grad).  This is a stop-gradient signal —
            # we only learn log_α; main net is not affected.
            with th.no_grad():
                per_agent_H = -(
                    actions_pdf * th.log(actions_pdf.clamp(min=1e-10))
                ).sum(dim=-1)                                              # [B, T, N]
                # Mask + average over valid timesteps + batch
                # mask is [B, T-1, 1]; pad with last-step's mask for [B, T, 1]
                mask_full = th.cat([mask, mask[:, -1:, :]], dim=1)
                denom = mask_full.sum(dim=(0, 1)).clamp(min=1.0).squeeze(-1)
                H_mean_per_agent = (per_agent_H * mask_full).sum(dim=(0, 1)) / denom
                # [N]
            # Lagrangian:  L_α = -log_α · (H_target - H_actual)
            alpha_loss = -(
                self.log_alpha * (self.target_entropy - H_mean_per_agent.detach())
            ).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        # Push α to the action selector for the next rollout.
        self._push_alpha_to_selector()

        # ═══════════════════════════════════════════════════════
        # 10. Target network updates
        # ═══════════════════════════════════════════════════════
        tau = self.args.target_update_interval_or_tau
        if tau > 1:
            if (episode_num - self.last_target_update_episode) / tau >= 1.0:
                self._update_targets_hard()
                self.last_target_update_episode = episode_num
        else:
            self._update_targets_soft(tau)

        # ═══════════════════════════════════════════════════════
        # 11. Logging
        # ═══════════════════════════════════════════════════════
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self._log_train_stats(
                t_env=t_env,
                loss_main=loss_main, L_td=L_td, L_beta=L_beta,
                grad_norm=grad_norm,
                mac_out=mac_out, chosen_Z_mixed=chosen_Z_mixed,
                targets_per_q=targets_per_q,
                target_entropy=target_entropy,
                alpha_vec=alpha_vec,
                synergy_loss=synergy_loss_for_log,
                synergy_bonus_mean=synergy_bonus_mean,
                slow_role_loss=slow_role_loss_for_log,
                slow_role_top1=slow_role_top1,
                sync_loss=sync_loss_for_log,
                mask=mask,
            )
            self.log_stats_t = t_env

    # ═══════════════════════════════════════════════════════════════════
    # Phase 4 — synergy
    # ═══════════════════════════════════════════════════════════════════

    def _apply_synergy_bonus(
        self, states, actions, mask,
        targets_per_q, rewards, terminated, target_entropy, target_Z_tot,
    ):
        """Train the synergy estimator and shape the per-step reward.

        Returns
        -------
        synergy_train_loss : scalar (for backward on synergy_optimizer)
        bonus_mean         : float (for logging)
        targets_per_q_new  : [B, T-1, K]  rebuilt TD(λ) target with shaped reward
        """
        bs = states.shape[0]
        T_minus_1 = mask.shape[1]
        device = states.device
        N = self.n_agents
        A = self.n_actions

        # Build flat tensors at the (B, T-1) timestep granularity.
        states_flat = states[:, :-1].reshape(bs * T_minus_1, -1)            # [F, S]
        actions_oh  = F.one_hot(actions.squeeze(-1).long(), num_classes=A).float()
        # actions_oh: [B, T-1, N, A]
        actions_oh_flat = actions_oh.reshape(bs * T_minus_1, N, A)          # [F, N, A]

        # G is the team return estimate (mean across quantiles of the TD(λ) target).
        G_flat = targets_per_q.mean(dim=-1, keepdim=True).reshape(bs * T_minus_1, 1).detach()
        # [F, 1]

        # Sample a triple (i, j, k) per timestep — i ≠ j, k chosen as the
        # agent we compare against (can be any one; uniform).  This Monte-Carlo
        # subsamples the O(N²) pair grid; the discriminator still sees every
        # pair eventually because the sampling is fresh per train step.
        i_idx = th.randint(0, N, (bs * T_minus_1, 1), device=device)
        # Sample j ≠ i via rejection-and-shift
        j_idx = th.randint(0, N - 1, (bs * T_minus_1, 1), device=device)
        j_idx = th.where(j_idx >= i_idx, j_idx + 1, j_idx)
        k_idx = th.randint(0, N, (bs * T_minus_1, 1), device=device)

        synergy_per_sample, train_loss = self.synergy_estimator.compute(
            states_flat, actions_oh_flat, G_flat,
            i_idx, j_idx, k_idx,
            n_negatives=self.synergy_n_neg,
        )
        # synergy_per_sample: [F, 1]  (broadcast scalar lower bound; we
        # additionally weight per-agent below).

        # Per-step, per-agent synergy attribution (averaged over j ≠ i).
        # For tractability we use the scalar synergy lower bound shared
        # across agents (all agents at this step receive the same bonus
        # proportional to S̃).  This is the conservative aggregation; a
        # finer version would compute S̃_{ij} for every pair, at O(N²) cost.
        S_clip = synergy_per_sample.clamp(min=0.0, max=self.synergy_clip)
        # [F, 1]  →  [B, T-1, 1]
        S_step = S_clip.reshape(bs, T_minus_1, 1).detach()

        # Train discriminator (gradient-isolated).
        self.synergy_optimizer.zero_grad()
        train_loss.backward()
        th.nn.utils.clip_grad_norm_(
            self.synergy_estimator.parameters(), self.args.grad_norm_clip,
        )
        self.synergy_optimizer.step()

        # ── Reward shaping  r̃_t = r_t + η · S̃_t ──
        rewards_shaped = rewards + self.synergy_eta * S_step
        # Rebuild TD(λ) targets with shaped rewards.  Same target_Z_tot, same
        # target_entropy, only `rewards_shaped` differs.
        with th.no_grad():
            targets_per_q_new = build_td_lambda_targets(
                rewards_shaped, terminated, mask,
                target_Z_tot, self.n_agents,
                self.args.gamma, self.args.td_lambda,
                target_entropy=target_entropy,
            )

        bonus_mean = float((self.synergy_eta * S_step).mean().detach().item())
        return train_loss.detach(), bonus_mean, targets_per_q_new

    # ═══════════════════════════════════════════════════════════════════
    # Phase 5a — slow-role
    # ═══════════════════════════════════════════════════════════════════

    def _compute_slow_role_loss(self, hidden_history, mask):
        """Future-conditioned InfoNCE on per-agent latents.

        hidden_history: [B, T, N, D]  (post-GRU agent latent at each step,
                                        already detached when captured)
        Returns scalar loss + top-1 accuracy diagnostic.
        """
        B, T, N, D = hidden_history.shape
        T_h = self.slow_role_horizon
        if T <= T_h:
            return hidden_history.new_zeros(()), 0.0

        # past = h_t, future = h_{t + T_h}; t in [0, T - T_h)
        past = hidden_history[:, :-T_h]                                    # [B, T-Th, N, D]
        future = hidden_history[:, T_h:]                                   # [B, T-Th, N, D]
        # Flatten time into batch:
        T_eff = T - T_h
        past_flat = past.reshape(B * T_eff, N, D)
        future_flat = future.reshape(B * T_eff, N, D)

        loss, stats = self.slow_role(past_flat, future_flat, return_stats=True)
        return loss, float(stats["slow_role_top1"])

    # ═══════════════════════════════════════════════════════════════════
    # Phase 5b — multi-timescale sync
    # ═══════════════════════════════════════════════════════════════════

    def _compute_sync_loss(self, hidden_history):
        """Predict each teammate's EMA-filtered latent at fast and slow bands.

        hidden_history: [B, T, N, D]
        """
        B, T, N, D = hidden_history.shape
        device = hidden_history.device

        # Compute EMAs along the time axis.  τ_fast = 0.95 → fast band
        # (~ 20-step memory); τ_slow = 0.99 → slow band (~ 100-step memory).
        ema_fast = th.zeros_like(hidden_history)
        ema_slow = th.zeros_like(hidden_history)
        ef = hidden_history[:, 0]
        es = hidden_history[:, 0]
        ema_fast[:, 0] = ef
        ema_slow[:, 0] = es
        for t in range(1, T):
            ef = self.sync_tau_fast * ef + (1.0 - self.sync_tau_fast) * hidden_history[:, t]
            es = self.sync_tau_slow * es + (1.0 - self.sync_tau_slow) * hidden_history[:, t]
            ema_fast[:, t] = ef
            ema_slow[:, t] = es

        loss = self.sync_predictor(hidden_history, [ema_fast, ema_slow])
        return loss

    # ═══════════════════════════════════════════════════════════════════
    # Logging
    # ═══════════════════════════════════════════════════════════════════

    def _log_train_stats(self, *, t_env, loss_main, L_td, L_beta, grad_norm,
                         mac_out, chosen_Z_mixed, targets_per_q,
                         target_entropy, alpha_vec,
                         synergy_loss, synergy_bonus_mean,
                         slow_role_loss, slow_role_top1, sync_loss,
                         mask):
        mask_elems = mask.sum().item()
        log = self.logger.log_stat
        log("loss",       loss_main.item(), t_env)
        log("loss_td",    L_td.item(),      t_env)
        log("loss_beta",  L_beta.item(),    t_env)
        log("grad_norm",
            grad_norm.item() if hasattr(grad_norm, "item") else grad_norm, t_env)

        log("agent_q_mean", mac_out.mean().item(), t_env)
        log("agent_q_std",  mac_out.std().item(),  t_env)
        log("q_taken_mean",
            (chosen_Z_mixed.mean(dim=-1, keepdim=True) * mask).sum().item()
            / max(mask_elems, 1.0), t_env)
        log("target_mean",
            (targets_per_q.mean(dim=-1, keepdim=True) * mask).sum().item()
            / max(mask_elems, 1.0), t_env)

        # Distributional diagnostics.
        q_low  = chosen_Z_mixed[..., 0].mean().item()
        q_high = chosen_Z_mixed[..., -1].mean().item()
        log("q_quantile_low",    q_low,             t_env)
        log("q_quantile_high",   q_high,            t_env)
        log("q_quantile_spread", q_high - q_low,    t_env)

        log("entropy",      target_entropy.mean().item(),       t_env)
        log("entropy_coef", self.entropy_coef,                  t_env)
        log("cvar_beta",    self._current_cvar_beta(t_env),     t_env)

        # Per-agent α.
        log("alpha_mean", float(alpha_vec.mean().item()), t_env)
        log("alpha_std",  float(alpha_vec.std().item()),  t_env)
        for i in range(self.n_agents):
            log(f"alpha_agent_{i}", float(alpha_vec[i].item()), t_env)

        if self.use_synergy_bonus:
            log("synergy_loss",       synergy_loss,       t_env)
            log("synergy_bonus_mean", synergy_bonus_mean, t_env)
        if self.use_slow_role:
            log("slow_role_loss", slow_role_loss, t_env)
            log("slow_role_top1", slow_role_top1, t_env)
        if self.use_sync_loss:
            log("sync_loss", sync_loss, t_env)

    # ═══════════════════════════════════════════════════════════════════
    # Target / device / save / load
    # ═══════════════════════════════════════════════════════════════════

    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for tp, p in zip(self.target_mac.parameters(), self.mac.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)
        for tp, p in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            tp.data.copy_(tp.data * (1.0 - tau) + p.data * tau)

    def cuda(self):
        device = getattr(self.args, "device", "cuda")
        self.mac.cuda()
        self.target_mac.cuda()
        self.mixer.to(device)
        self.target_mixer.to(device)
        self.taus = self.taus.to(device)
        if self.use_dual_alpha:
            self.log_alpha.data = self.log_alpha.data.to(device)
        if self.use_synergy_bonus:
            self.synergy_estimator.to(device)
        if self.use_slow_role:
            self.slow_role.to(device)
        if self.use_sync_loss:
            self.sync_predictor.to(device)

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.mixer.state_dict(),         "{}/mixer.th".format(path))
        th.save(self.main_optimizer.state_dict(),"{}/opt.th".format(path))
        if self.use_dual_alpha:
            th.save({"log_alpha": self.log_alpha.detach().cpu()},
                    "{}/log_alpha.th".format(path))
            th.save(self.alpha_optimizer.state_dict(),
                    "{}/alpha_opt.th".format(path))
        if self.use_synergy_bonus:
            th.save(self.synergy_estimator.state_dict(),
                    "{}/synergy.th".format(path))
            th.save(self.synergy_optimizer.state_dict(),
                    "{}/synergy_opt.th".format(path))
        if self.use_slow_role:
            th.save(self.slow_role.state_dict(),
                    "{}/slow_role.th".format(path))
            th.save(self.slow_role_optimizer.state_dict(),
                    "{}/slow_role_opt.th".format(path))
        if self.use_sync_loss:
            th.save(self.sync_predictor.state_dict(),
                    "{}/sync.th".format(path))
            th.save(self.sync_optimizer.state_dict(),
                    "{}/sync_opt.th".format(path))

    def load_models(self, path):
        import os
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        self.mixer.load_state_dict(
            th.load("{}/mixer.th".format(path),
                    map_location=lambda s, l: s)
        )
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.main_optimizer.load_state_dict(
            th.load("{}/opt.th".format(path),
                    map_location=lambda s, l: s)
        )
        if self.use_dual_alpha and os.path.exists("{}/log_alpha.th".format(path)):
            blob = th.load("{}/log_alpha.th".format(path),
                           map_location=lambda s, l: s)
            self.log_alpha.data.copy_(blob["log_alpha"])
            self.alpha_optimizer.load_state_dict(
                th.load("{}/alpha_opt.th".format(path),
                        map_location=lambda s, l: s)
            )
        if self.use_synergy_bonus and os.path.exists("{}/synergy.th".format(path)):
            self.synergy_estimator.load_state_dict(
                th.load("{}/synergy.th".format(path),
                        map_location=lambda s, l: s)
            )
            self.synergy_optimizer.load_state_dict(
                th.load("{}/synergy_opt.th".format(path),
                        map_location=lambda s, l: s)
            )
        if self.use_slow_role and os.path.exists("{}/slow_role.th".format(path)):
            self.slow_role.load_state_dict(
                th.load("{}/slow_role.th".format(path),
                        map_location=lambda s, l: s)
            )
            self.slow_role_optimizer.load_state_dict(
                th.load("{}/slow_role_opt.th".format(path),
                        map_location=lambda s, l: s)
            )
        if self.use_sync_loss and os.path.exists("{}/sync.th".format(path)):
            self.sync_predictor.load_state_dict(
                th.load("{}/sync.th".format(path),
                        map_location=lambda s, l: s)
            )
            self.sync_optimizer.load_state_dict(
                th.load("{}/sync_opt.th".format(path),
                        map_location=lambda s, l: s)
            )
