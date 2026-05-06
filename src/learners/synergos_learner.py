"""SynergosLearner — Phase 1 of the SYNERGOS algorithm proposal.

Phase 1 = Distributional Soft-QMIX with CVaR-Soft exploration on top of the
existing CASVD backbone (GAT encoder + Soft-QMIX VDN-sum mixer + entropy-
regularised TD(λ)).  The single experimental variable in this phase is the
value-head representation:

    Soft-QMIX (CASVD)   :  scalar  Q(s, a)         + scalar TD loss
    SYNERGOS Phase 1    :  K-quantile  Z(s, a; τ)  + quantile Huber loss
                            CVaR_β-Soft policy for sampling

Everything else — GAT trunk, hard target updates, Soft-QMIX β-loss, scalar
α (entropy_coef), TD(λ) — is preserved from CASVD so that any observed gain
isolates the contribution of distributional value learning + risk-sensitive
sampling.

Phases 2-5 (sheaf mixer, synergy bonus, slow-role, dual α, sync) are deferred
to subsequent learners; this file deliberately stays minimal.
"""
import copy

import torch as th
from torch.optim import Adam

from modules.mixers.dist_soft_mix import DistSoftMixer
from utils.rl_utils import build_td_lambda_targets


class SynergosLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.K = int(getattr(args, "n_quantiles", 8))

        # QR-DQN fixed quantile fractions  τ_k = (2k − 1) / (2K),  k = 1 .. K.
        # Stored on CPU; moved to device in `cuda()` so the loss broadcasts
        # without per-step host→device copies.
        self.taus = th.tensor(
            [(2 * k - 1) / (2.0 * self.K) for k in range(1, self.K + 1)],
            dtype=th.float32,
        )

        # Target network and distributional Soft mixer.
        self.target_mac = copy.deepcopy(self.mac)
        self.mixer = DistSoftMixer(args)
        self.target_mixer = copy.deepcopy(self.mixer)
        self.mac.set_mixer(self.mixer)

        # Soft-QMIX entropy coef (scalar in Phase 1).
        self.entropy_coef = float(getattr(args, "entropy_coef", 0.03))

        # CVaR β annealing (Phase 1 default: 1.0 → 0.25 over the first 1 M steps,
        # so all early exploration is mean-Q-based and risk-sensitivity fades in).
        self.cvar_beta_start = float(getattr(args, "cvar_beta_start", 1.0))
        self.cvar_beta_end   = float(getattr(args, "cvar_beta_end",   0.25))
        self.cvar_anneal_start = int(getattr(args, "cvar_anneal_start", 0))
        self.cvar_anneal_end   = int(getattr(args, "cvar_anneal_end",   1_000_000))

        # Quantile Huber κ (Dabney et al. 2018 use 1.0).
        self.huber_kappa = float(getattr(args, "huber_kappa", 1.0))

        opt_eps = getattr(args, "optimizer_epsilon", 1e-7)
        self.params = list(self.mac.parameters()) + list(self.mixer.parameters())
        self.optimizer = Adam(self.params, lr=args.lr, eps=opt_eps)

        self.last_target_update_episode = 0
        self.log_stats_t = -self.args.learner_log_interval - 1

        # Push initial scalar α to the action selector so rollout uses the
        # same temperature as the target computation from step 0.
        if hasattr(self.mac, "set_alpha"):
            self.mac.set_alpha(self.entropy_coef)

    # ───────────────────────────────────────────────────────────────────
    # Helpers
    # ───────────────────────────────────────────────────────────────────

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
        """CVaR_β over the last (quantile) dim.  z: [..., K] → [...]."""
        K = z.shape[-1]
        m = max(1, int(round(float(beta) * K)))
        z_sorted, _ = th.sort(z, dim=-1)
        return z_sorted[..., :m].mean(dim=-1)

    def _quantile_huber_loss(self, pred, target, mask):
        """Quantile Huber loss with QR-DQN's fixed τ.

        pred:   [B, T-1, K]  predicted quantile values  (Z_tot at chosen action)
        target: [B, T-1, K]  target  quantile values     (TD(λ) per quantile)
        mask:   [B, T-1, 1]  valid-step mask
        """
        K = pred.shape[-1]
        kappa = self.huber_kappa

        pred_exp   = pred.unsqueeze(-1)        # [B, T-1, K, 1]
        target_exp = target.unsqueeze(-2)      # [B, T-1, 1, K]
        delta = target_exp - pred_exp          # [B, T-1, K_pred, K_target]

        abs_delta = delta.abs()
        huber = th.where(
            abs_delta <= kappa,
            0.5 * delta.pow(2),
            kappa * (abs_delta - 0.5 * kappa),
        )
        # Ensure τ is on the right device.
        if self.taus.device != delta.device:
            self.taus = self.taus.to(delta.device)
        taus = self.taus.view(1, 1, K, 1)
        indicator = (delta < 0.0).float()
        rho = (taus - indicator).abs() * huber / kappa
        # Sum over target K, mean over predicted K, then mean across batch×time
        loss_per_step = rho.sum(dim=-1).mean(dim=-1, keepdim=True)  # [B, T-1, 1]
        loss = (loss_per_step * mask).sum() / mask.sum().clamp(min=1.0)
        return loss

    # ───────────────────────────────────────────────────────────────────
    # Train step
    # ───────────────────────────────────────────────────────────────────

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
        K = self.K

        # ═══════════════════════════════════════════════════════
        # 1.  Online forward — distributional Z [B, T, N, A, K]
        # ═══════════════════════════════════════════════════════
        self.mac.init_hidden(bs)
        mac_out = []
        for t in range(T):
            z_t = self.mac.forward(batch, t)               # [B, N, A, K]
            mac_out.append(z_t)
        mac_out = th.stack(mac_out, dim=1)                 # [B, T, N, A, K]
        mac_out = self.mixer.func_g_dist(mac_out, states, t_env)

        # Chosen-action distribution: gather across action dim.
        # actions is [B, T-1, N, 1]; expand a quantile dim and gather.
        actions_exp = actions.unsqueeze(-1).expand(-1, -1, -1, -1, K)   # [B, T-1, N, 1, K]
        chosen_Z = th.gather(mac_out[:, :-1], dim=3, index=actions_exp).squeeze(3)
        # chosen_Z: [B, T-1, N, K]

        # ═══════════════════════════════════════════════════════
        # 2.  Target — distributional Z under target net + soft policy
        # ═══════════════════════════════════════════════════════
        with th.no_grad():
            self.target_mac.init_hidden(bs)
            target_mac_out = []
            for t in range(T):
                z_t = self.target_mac.forward(batch, t)
                target_mac_out.append(z_t)
            target_mac_out = th.stack(target_mac_out, dim=1)
            target_mac_out = self.target_mixer.func_g_dist(target_mac_out, states, t_env)
            # [B, T, N, A, K]

            # Soft policy from ONLINE net (Double-Q):  apply func_f, take CVaR_β.
            mac_out_for_policy = mac_out.clone().detach()
            mac_out_for_policy = self.mixer.func_f_dist(mac_out_for_policy, states, t_env)
            beta = self._current_cvar_beta(t_env)
            v_beta = self._cvar(mac_out_for_policy, beta)              # [B, T, N, A]
            v_beta = v_beta.masked_fill(avail_actions == 0, -1e9)
            alpha_scalar = self.entropy_coef
            logits = v_beta / alpha_scalar
            actions_pdf = th.softmax(logits, dim=-1)                   # [B, T, N, A]

            # Sample a' via the inverse-CDF trick (numerically stable).
            rand_idx = th.rand(actions_pdf[..., :1].shape, device=actions_pdf.device)
            actions_cdf = th.cumsum(actions_pdf, dim=-1)
            rand_idx = th.clamp(rand_idx, 1e-6, 1 - 1e-6)
            picked_actions = th.searchsorted(actions_cdf, rand_idx)    # [B, T, N, 1]

            # Gather target Z at the sampled actions.
            picked_exp = picked_actions.unsqueeze(-1).expand(-1, -1, -1, -1, K)
            target_Z = th.gather(target_mac_out, dim=3, index=picked_exp).squeeze(3)
            # target_Z: [B, T, N, K]

            # Soft-QMIX entropy bonus  −α · log π(a' | s)  summed over agents.
            target_logp = th.log(actions_pdf + 1e-10)                  # [B, T, N, A]
            target_logp = th.gather(target_logp, dim=3, index=picked_actions).squeeze(3)
            # [B, T, N]
            target_entropy = -alpha_scalar * target_logp.sum(-1, keepdim=True)
            # [B, T, 1] — same bonus is added to every quantile target.

            # VDN-sum across agents per quantile.
            target_Z_tot = target_Z.sum(dim=2)                         # [B, T, K]

            # TD(λ) per quantile.  build_td_lambda_targets broadcasts over
            # the final dim, so we can pass [B, T, K] directly without folding.
            targets_per_q = build_td_lambda_targets(
                rewards, terminated, mask,
                target_Z_tot, self.n_agents,
                self.args.gamma, self.args.td_lambda,
                target_entropy=target_entropy,
            )                                                          # [B, T-1, K]

        # ═══════════════════════════════════════════════════════
        # 3.  Loss — quantile Huber + Soft-QMIX β loss (on the mean)
        # ═══════════════════════════════════════════════════════
        chosen_Z_mixed = chosen_Z.sum(dim=2)                           # [B, T-1, K]
        L_td = self._quantile_huber_loss(chosen_Z_mixed, targets_per_q, mask)

        # β-loss: keep func_f ≈ identity, computed on the per-quantile MEAN
        # (preserves the original CASVD interpretation of L_β as a soft
        # constraint that the affine func_f recovers Q_tot).
        chosen_Z_mean_per_agent = chosen_Z.mean(dim=-1)                # [B, T-1, N]
        chosen_Z_mixed_mean = chosen_Z_mixed.mean(dim=-1, keepdim=True)  # [B, T-1, 1]
        affine_aq = self.mixer.func_f(
            chosen_Z_mean_per_agent.detach(),
            states[:, :-1],
            t_env,
        )
        approx_error = chosen_Z_mixed_mean.detach() - affine_aq.sum(-1, keepdim=True)
        beta_error = 0.5 * approx_error.pow(2)
        L_beta = (beta_error * mask).sum() / mask.sum().clamp(min=1.0)

        loss = L_td + L_beta

        # ═══════════════════════════════════════════════════════
        # 4.  Optimise
        # ═══════════════════════════════════════════════════════
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimizer.step()

        # Push α to the selector (scalar α for Phase 1).
        if hasattr(self.mac, "set_alpha"):
            self.mac.set_alpha(self.entropy_coef)

        # ═══════════════════════════════════════════════════════
        # 5.  Target updates
        # ═══════════════════════════════════════════════════════
        tau = self.args.target_update_interval_or_tau
        if tau > 1:
            if (episode_num - self.last_target_update_episode) / tau >= 1.0:
                self._update_targets_hard()
                self.last_target_update_episode = episode_num
        else:
            self._update_targets_soft(tau)

        # ═══════════════════════════════════════════════════════
        # 6.  Logging
        # ═══════════════════════════════════════════════════════
        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            mask_elems = mask.sum().item()
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("loss_beta", L_beta.item(), t_env)
            self.logger.log_stat(
                "grad_norm",
                grad_norm.item() if hasattr(grad_norm, "item") else grad_norm,
                t_env,
            )

            self.logger.log_stat("agent_q_mean",  mac_out.mean().item(),       t_env)
            self.logger.log_stat("agent_q_std",   mac_out.std().item(),        t_env)
            self.logger.log_stat("q_taken_mean",
                                 (chosen_Z_mixed.mean(dim=-1, keepdim=True) * mask).sum().item()
                                 / max(mask_elems, 1.0),
                                 t_env)
            self.logger.log_stat("target_mean",
                                 (targets_per_q.mean(dim=-1, keepdim=True) * mask).sum().item()
                                 / max(mask_elems, 1.0),
                                 t_env)

            # Distributional diagnostics — capturing the actual shape of the
            # learned return distribution.
            #   q_quantile_spread  : top quantile − bottom quantile of Z_tot
            #                        at the chosen action; tracks aleatoric
            #                        variance the model has captured.
            #   q_quantile_low     : mean of bottom quantile (worst-case Q).
            #   q_quantile_high    : mean of top quantile    (best-case Q).
            q_low  = chosen_Z_mixed[..., 0].mean().item()
            q_high = chosen_Z_mixed[..., -1].mean().item()
            self.logger.log_stat("q_quantile_low",    q_low,             t_env)
            self.logger.log_stat("q_quantile_high",   q_high,            t_env)
            self.logger.log_stat("q_quantile_spread", q_high - q_low,    t_env)

            self.logger.log_stat("entropy",       target_entropy.mean().item(), t_env)
            self.logger.log_stat("entropy_coef",  self.entropy_coef,            t_env)
            self.logger.log_stat("cvar_beta",     self._current_cvar_beta(t_env), t_env)

            self.log_stats_t = t_env

    # ───────────────────────────────────────────────────────────────────
    # Target updates / device / save / load
    # ───────────────────────────────────────────────────────────────────

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

    def save_models(self, path):
        self.mac.save_models(path)
        th.save(self.mixer.state_dict(),     "{}/mixer.th".format(path))
        th.save(self.optimizer.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        self.target_mac.load_models(path)
        self.mixer.load_state_dict(
            th.load("{}/mixer.th".format(path),
                    map_location=lambda storage, loc: storage)
        )
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optimizer.load_state_dict(
            th.load("{}/opt.th".format(path),
                    map_location=lambda storage, loc: storage)
        )
