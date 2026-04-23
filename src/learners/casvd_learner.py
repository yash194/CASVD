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

        # ── α mode toggle ──
        # Modes:
        #   "scalar"           – all agents share α = entropy_coef (Run A baseline).
        #   "per_agent_fixed"  – hardcoded per-agent α from config (Run B premise test).
        #   "adaptive"         – α driven by InfoNCE coord_signal EMA (legacy path;
        #                        code kept intact for later diagnostics).
        #   "q_spread_adaptive"– NEW (Component 2).  Per-agent α derived from the
        #                        current Q-value spread: confident agent (wide spread,
        #                        clear best action) gets low α and commits; uncertain
        #                        agent (flat Q) gets high α and hedges.  Uses only
        #                        signals already computed by the learner — no extra
        #                        network, no second optimiser.
        default_mode = "adaptive" if getattr(args, "use_adaptive_alpha", False) else "scalar"
        self.alpha_mode = str(getattr(args, "alpha_mode", default_mode)).lower()
        assert self.alpha_mode in (
            "scalar", "per_agent_fixed", "adaptive", "q_spread_adaptive"
        ), (
            "alpha_mode must be one of 'scalar', 'per_agent_fixed', 'adaptive', "
            f"'q_spread_adaptive'; got {self.alpha_mode!r}"
        )

        if self.alpha_mode == "per_agent_fixed":
            per_agent_alphas = list(getattr(args, "per_agent_alphas", []))
            assert len(per_agent_alphas) == self.n_agents, (
                f"per_agent_alphas must have length n_agents={self.n_agents}, "
                f"got {len(per_agent_alphas)}: {per_agent_alphas}"
            )
            assert all(a > 0 for a in per_agent_alphas), (
                f"per_agent_alphas must be strictly positive; got {per_agent_alphas}"
            )
            self._alpha_vec_cpu = th.tensor(per_agent_alphas, dtype=th.float32)
        else:
            # scalar and adaptive both initialise as a uniform vector at
            # entropy_coef.  adaptive mode mutates this vector on the fly
            # using coord_signals; scalar leaves it untouched.
            self._alpha_vec_cpu = th.full(
                (self.n_agents,), float(self.entropy_coef), dtype=th.float32
            )
        self._alpha_vec = None  # lazy device placement

        # ── Per-agent adaptive alpha (off by default) ──
        self.use_adaptive_alpha = (self.alpha_mode == "adaptive")
        self._coord_signals = None
        self._coord_warmup = True   # first batch seeds directly, no EMA blending
        # τ = 0.95 → time constant ~20 training steps ≈ 0.2 % of a 10 M-step run.
        # Fast enough that early-training coordination changes actually move the
        # signal; slow enough that per-batch sampling noise averages out.
        self.coord_signal_ema_tau = getattr(args, "coord_signal_ema_tau", 0.95)

        # ── Component 2: Q-spread sensor state ──
        # EMA of per-agent Q-spread (max_a Q - min_a Q over avail actions).
        # Populated each train step when alpha_mode == "q_spread_adaptive".
        self._q_spread_ema = None
        self.q_spread_ema_tau = getattr(args, "q_spread_ema_tau", 0.99)
        # Confidence ratio clamp — bounds α to
        # [entropy_coef / conf_max, entropy_coef / conf_min].
        # With defaults 0.3–3.0 → α ∈ [α_mean/3, α_mean·3.33].  Prevents
        # degenerate greedy / uniform agents when spreads are extreme.
        self.q_spread_conf_min = getattr(args, "q_spread_conf_min", 0.3)
        self.q_spread_conf_max = getattr(args, "q_spread_conf_max", 3.0)

        # ── Component 3: curriculum warmup for α heterogeneity ──
        # Ramp 0→1 applied to (α_het − α_mean) so training starts as pure
        # scalar-α Soft-QMIX and phases in per-agent heterogeneity over
        # [alpha_warmup_start, alpha_warmup_end].  Prevents the early-training
        # coordination tax that caused Run B's stalling attractor.
        self.alpha_warmup_start = int(getattr(args, "alpha_warmup_start", 2_000_000))
        self.alpha_warmup_end   = int(getattr(args, "alpha_warmup_end",   4_000_000))
        assert self.alpha_warmup_end >= self.alpha_warmup_start >= 0, (
            f"alpha_warmup_end ({self.alpha_warmup_end}) must be ≥ "
            f"alpha_warmup_start ({self.alpha_warmup_start}) ≥ 0"
        )

        # ── InfoNCE coordination sensor (off by default) ──
        self.lgdd_enabled = getattr(args, "lgdd_enabled", False)
        self.infonce_n_negatives = getattr(args, "infonce_n_negatives", 15)
        self.infonce_temperature = getattr(args, "infonce_temperature", 0.1)

        self.dynamics_predictor = None
        self.dynamics_params = []

        if self.lgdd_enabled:
            from modules.predictors import InfoNCEPredictor

            # Stage C: identity-conditioned predictor.
            # Passing n_agents enables appending a one-hot agent ID to
            # h_i before the projector.  Breaks shared-predictor symmetry
            # even when local_summary values are similar across agents.
            self.dynamics_predictor = InfoNCEPredictor(
                args.hidden_dim,
                n_agents=self.n_agents,
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

        # Seed the action selector with the initial α vector so that
        # rollout and target computation use the same α from step 0.
        # (For adaptive mode this is overwritten after each train step
        # once coord_signals update.)
        self.mac.set_alpha(self._alpha_vec_cpu.clone())

    def _alpha_ramp(self, t_env):
        """Component 3 curriculum: linear ramp 0→1 over the warmup window.

        Before `alpha_warmup_start` → 0 (α is uniform α_mean).
        After  `alpha_warmup_end`   → 1 (full heterogeneous α).
        Between → linear interpolation.
        """
        if t_env is None:
            return 1.0
        start = self.alpha_warmup_start
        end   = self.alpha_warmup_end
        if t_env <= start:
            return 0.0
        if t_env >= end:
            return 1.0
        span = max(1, end - start)
        return float(t_env - start) / float(span)

    def _update_q_spread_ema(self, mac_out, avail_actions, mask):
        """Component 2: update per-agent Q-spread EMA from the current batch.

        Q-spread_i(b, t) = max_{a ∈ avail} Q_i(s_{b,t}, a)
                         − min_{a ∈ avail} Q_i(s_{b,t}, a)

        Wide spread → agent has a clearly best action → should commit (low α).
        Narrow spread → agent is uncertain → should hedge (high α).

        Averaged over valid timesteps (with mask) and over agents that actually
        have at least one available action in that step (drops dead-agent noise).
        Feeds a per-agent EMA with τ = `q_spread_ema_tau`.
        """
        avail_float = avail_actions.float()
        # Mask unavailable actions so they don't contaminate max / min
        q_for_max = mac_out.masked_fill(avail_actions == 0, float("-inf"))
        q_for_min = mac_out.masked_fill(avail_actions == 0, float("inf"))
        q_spread = q_for_max.max(dim=-1).values - q_for_min.min(dim=-1).values  # [B, T, N]

        # Slice to training timesteps (mask is [B, T-1, 1])
        q_spread = q_spread[:, :-1]                                              # [B, T-1, N]
        # Any agent with NO avail actions at (b, t) → spread is ±inf; zero it out
        q_spread = th.nan_to_num(q_spread, nan=0.0, posinf=0.0, neginf=0.0)

        # Per-(b, t, i) validity: step is valid AND agent has avail actions
        has_avail = (avail_float[:, :-1].sum(dim=-1) > 0).float()                # [B, T-1, N]
        mask_bt_i = mask.expand_as(q_spread) * has_avail                         # [B, T-1, N]

        denom = mask_bt_i.sum(dim=(0, 1)).clamp(min=1.0)                         # [N]
        q_spread_per_agent = (q_spread * mask_bt_i).sum(dim=(0, 1)) / denom      # [N]
        q_spread_per_agent = q_spread_per_agent.detach()

        if self._q_spread_ema is None or self._q_spread_ema.device != q_spread_per_agent.device:
            self._q_spread_ema = q_spread_per_agent.clone()
        else:
            tau = self.q_spread_ema_tau
            self._q_spread_ema = tau * self._q_spread_ema + (1.0 - tau) * q_spread_per_agent

    def _get_alpha_vec(self, device, t_env=None):
        """Return per-agent α as a [n_agents] tensor on `device`.

        Computes a heterogeneous α_het based on `alpha_mode`, then blends
        it against the uniform α_mean via the Component-3 curriculum ramp:

            α_effective = α_mean + ramp(t_env) · (α_het − α_mean)

        At t_env ≤ alpha_warmup_start ⇒ α = α_mean (pure scalar Soft-QMIX).
        At t_env ≥ alpha_warmup_end   ⇒ α = α_het  (full heterogeneity).

        Modes:
          - scalar            → α_het = α_mean (ramp irrelevant)
          - per_agent_fixed   → α_het = hardcoded config vector
          - adaptive          → α_het from coord_signal EMA (legacy, intact)
          - q_spread_adaptive → α_het = α_mean / confidence_i where
                                 confidence_i = clamp(spread_i / spread_mean,
                                                      conf_min, conf_max).
                                 Confident agent ⇒ α_i < α_mean (commits).
                                 Uncertain agent ⇒ α_i > α_mean (hedges).
        """
        if self._alpha_vec is None or self._alpha_vec.device != device:
            self._alpha_vec = self._alpha_vec_cpu.to(device)

        # ── Select the heterogeneous target α for this mode ──
        if self.alpha_mode == "q_spread_adaptive" and self._q_spread_ema is not None:
            spread = self._q_spread_ema.to(device)
            mean_spread = spread.mean().clamp(min=1e-6)
            confidence = (spread / mean_spread).clamp(
                self.q_spread_conf_min, self.q_spread_conf_max
            )
            alpha_het = float(self.entropy_coef) / confidence
        elif self.alpha_mode == "adaptive" and self._coord_signals is not None:
            # Legacy InfoNCE-driven path — kept intact for later use.
            coord = self._coord_signals.to(device).clamp(0.0, 1.0)
            alpha_het = float(self.entropy_coef) * (0.5 + coord)
        else:
            # scalar / per_agent_fixed / pre-EMA q_spread_adaptive
            alpha_het = self._alpha_vec

        # ── Component 3: curriculum blend towards α_mean ──
        ramp = self._alpha_ramp(t_env)
        if ramp >= 1.0:
            return alpha_het
        if ramp <= 0.0:
            # Full uniform α_mean — reuse or build on-device tensor
            return th.full_like(self._alpha_vec, float(self.entropy_coef))

        alpha_mean_val = float(self.entropy_coef)
        alpha_vec = alpha_mean_val + ramp * (alpha_het - alpha_mean_val)
        return alpha_vec

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

    def _compute_infonce(self, all_hidden, mask, device,
                          all_local=None, all_team=None):
        """Per-agent InfoNCE coord signal — Stage B (pre-TeamGAT signals).

        Stage B of the UPMI fix:  both predictor INPUT and TARGET are
        taken from local_summary (pre-TeamGAT), not the post-GAT hidden
        state.  Empirically, `delta_local_cross_agent_cos ≈ 0.04`
        (~orthogonal) while `delta_h_cross_agent_cos ≈ 0.46` — TeamGAT
        is the homogeniser, and pre-TeamGAT deltas are the per-agent
        signal the adaptive-α pipeline needs.

        Setup:
            INPUT (to predictor):  local_summary_i(t)        cos~0.76
            TARGET (positive):     Δlocal_summary_j(t)       cos~0.04
            NEGATIVE:              Δlocal_summary, cross-episode mean

        Rationale (Stage A failure mode):  with post-GAT targets
        `h_cross_agent_cos = 0.81`, the predictor's output W(h_i) is
        nearly identical across agents regardless of pairwise vs mean
        aggregation.  Per-pair variance existed (std 0.6) but the
        per-agent aggregate was uniform (coord_signal_std ≈ 0.004).
        Pre-TeamGAT signals break the shared-encoder symmetry at
        BOTH the input and target sides.

        Pairwise structure preserved from Stage A:  compute InfoNCE
        separately for each (agent i → teammate j) pair, then mean
        the LOSSES, keeping per-teammate variance from being washed
        out by pre-cosine averaging.

        Args:
            all_hidden: list of [B, n_agents, D] hidden tensors, length T.
                        Kept for backward-compat cross-agent diagnostics
                        (h_cross_agent_cos), but NOT used for the loss.
            mask:       [B, T-1, 1] valid-timestep mask.
            device:     target device.
            all_local:  list of [B, n_agents, D] pre-TeamGAT local_summary
                        tensors (detached clones). REQUIRED for Stage B.
            all_team:   list of [B, n_agents, D] post-TeamGAT team_summary.
                        Diagnostic only.

        Returns:
            infonce_loss:       scalar, mean InfoNCE loss for backward.
            coord_signal_batch: [n_agents] normalised per-agent signal.
            per_agent_loss:     [B, n_agents] time-masked per-agent loss.
            diag:               dict of diagnostic stats.
        """
        assert all_local is not None and len(all_local) > 0, (
            "Stage B requires pre-TeamGAT local_summary — "
            "train() must call forward_with_latents with lgdd_enabled."
        )

        # Stage B: PRIMARY tensor is now local_summary (pre-TeamGAT).
        all_loc = th.stack(all_local, dim=0)            # [T, B, N, D]
        T_size, B_size, N, D = all_loc.shape
        K = self.infonce_n_negatives

        # Keep all_h available for diagnostics only.
        all_h = th.stack(all_hidden, dim=0)              # [T, B, N, D]

        # ── Δlocal_summary — the Stage B target ─────────────────────
        # Δlocal_j(t) = local_summary_j(t+1) − local_summary_j(t).
        # Empirical cross_agent_cos ≈ 0.04 (near-orthogonal) — the
        # per-agent signal has been hiding here the whole time.
        delta_loc = all_loc[1:] - all_loc[:-1]           # [T-1, B, N, D]
        T_delta = T_size - 1

        # Also compute Δh for comparison diagnostic only (not used in loss).
        delta_h = all_h[1:] - all_h[:-1]                 # [T-1, B, N, D]

        # Global team delta for cross-episode negatives (based on local).
        g_all_delta = delta_loc.mean(dim=2)              # [T-1, B, D]

        # Cross-episode negative sampling (same as before).
        if B_size >= 2:
            shifts = th.randint(1, B_size, (T_delta, B_size, K), device=device)
        else:
            shifts = th.zeros((T_delta, B_size, K), dtype=th.long, device=device)
        neg_t = th.randint(0, T_delta, (T_delta, B_size, K), device=device)
        b_arange = th.arange(B_size, device=device).view(1, B_size, 1)
        neg_b = (b_arange + shifts) % B_size             # [T-1, B, K]
        g_neg_all = g_all_delta[neg_t, neg_b]            # [T-1, B, K, D]

        # ── Per-teammate targets — from Δlocal ────────────────────
        if N > 1:
            teammate_idx = th.tensor(
                [[j for j in range(N) if j != i] for i in range(N)],
                dtype=th.long, device=device,
            )                                            # [N, N-1]
            teammate_deltas = delta_loc[:, :, teammate_idx]
            # shape [T-1, B, N, N-1, D]:
            # teammate_deltas[τ, b, i, k] = Δlocal_{teammate_idx[i,k]} at (τ,b)
            n_teammates = N - 1
        else:
            teammate_deltas = delta_loc.unsqueeze(3)     # [T-1, B, N, 1, D]
            n_teammates = 1

        # ── Flatten (T-1, B) → F for the predictor ───────────────
        # Predictor INPUT is local_summary_i(t) (NOT its delta).
        F_dim = T_delta * B_size
        h_i_all          = all_loc[:T_size - 1]          # [T-1, B, N, D]
        h_i_flat         = h_i_all.reshape(F_dim, N, D)  # [F, N, D]
        g_neg_flat       = g_neg_all.reshape(F_dim, K, D)
        teammate_deltas_flat = teammate_deltas.reshape(
            F_dim, N, n_teammates, D
        )                                                # [F, N, N-1, D]

        # ── Pairwise InfoNCE: one loss per (i, k) teammate slot ───
        # Loop over k=0..N-2.  Only 4 iterations for SMACv2 Protoss 5v5.
        per_teammate_losses = []
        per_teammate_stats  = []
        for k in range(n_teammates):
            g_pos_k = teammate_deltas_flat[:, :, k, :]       # [F, N, D]
            loss_k, stats_k = self.dynamics_predictor(
                h_i_flat, g_pos_k, g_neg_flat, return_stats=True
            )                                                # [F, N]
            per_teammate_losses.append(loss_k)
            per_teammate_stats.append(stats_k)

        per_pair_loss = th.stack(per_teammate_losses, dim=-1)  # [F, N, N-1]
        # Average over teammate slots  →  per-(f, i) loss
        loss_flat = per_pair_loss.mean(dim=-1)                 # [F, N]

        all_per_agent_loss = loss_flat.reshape(
            T_delta, B_size, N
        ).permute(1, 0, 2)                                     # [B, T-1, N]

        # Masked time-average → [B, N]
        mask_sq  = mask.squeeze(-1)                            # [B, T-1]
        mask_exp = mask_sq.unsqueeze(-1)                       # [B, T-1, 1]
        denom    = mask_exp.sum(dim=1).clamp(min=1.0)
        per_agent_loss = (all_per_agent_loss * mask_exp).sum(dim=1) / denom

        # Scalar loss for backward (mean over B and N)
        infonce_loss = per_agent_loss.mean()

        # Per-agent coordination signal in [0, 1]
        max_infonce = math.log(K + 1)
        coord_signal_batch = (
            per_agent_loss.mean(dim=0) / max_infonce
        ).clamp(0.0, 1.0)

        # ── Diagnostic stats ─────────────────────────────────────
        with th.no_grad():
            # Kept for comparison with previous runs (h-based stats):
            h_diversity     = all_h.std(dim=2).mean().item()
            delta_diversity = delta_h.std(dim=2).mean().item()
            delta_norm_mean = delta_h.norm(dim=-1).mean().item()

            # NEW: Stage-B-specific diagnostics on the actual signal
            # used by the predictor (local_summary and its delta).
            local_diversity       = all_loc.std(dim=2).mean().item()
            delta_local_diversity = delta_loc.std(dim=2).mean().item()
            delta_local_norm_mean = delta_loc.norm(dim=-1).mean().item()

            # ── Cross-agent cosine similarity (upstream diagnostic) ──
            # Stage B uses local_summary as input and Δlocal as target,
            # so `local_summary_cross_agent_cos` is now the INPUT-side
            # alignment metric (not h) and `delta_local_cross_agent_cos`
            # is the TARGET-side metric.  The h-based versions are kept
            # for cross-comparison with the pre-Stage-B runs.
            if N > 1:
                n_off = N * (N - 1)

                def _cross_agent_cos(x):
                    """Mean of off-diagonal cos(x_i, x_j).  x: [T, B, N, D]."""
                    x_n = F.normalize(x, dim=-1)
                    mat = th.einsum("tbid,tbjd->tbij", x_n, x_n)
                    off_sum = (
                        mat.sum(dim=(-2, -1))
                        - th.diagonal(mat, dim1=-2, dim2=-1).sum(dim=-1)
                    )
                    return (off_sum / n_off).mean().item()

                h_cross_agent_cos             = _cross_agent_cos(all_h)
                delta_cross_agent_cos         = _cross_agent_cos(delta_h)
                local_summary_cross_agent_cos = _cross_agent_cos(all_loc)
                delta_local_cross_agent_cos   = _cross_agent_cos(delta_loc)

                if all_team is not None and len(all_team) > 0:
                    all_team_stack = th.stack(all_team, dim=0)
                    team_summary_cross_agent_cos = _cross_agent_cos(all_team_stack)
                else:
                    team_summary_cross_agent_cos = -1.0
            else:
                h_cross_agent_cos = 1.0
                delta_cross_agent_cos = 1.0
                local_summary_cross_agent_cos = 1.0
                team_summary_cross_agent_cos = 1.0
                delta_local_cross_agent_cos = 1.0

            # raw_cos_sim: how close is local_summary_i to a representative
            # teammate's Δlocal BEFORE the predictor projects?  If low, task
            # is genuinely non-trivial.  Expected for Stage B: very low
            # (~0), because local_summary (cos 0.76) and Δlocal (cos 0.04)
            # live in roughly orthogonal parts of the embedding manifold.
            if n_teammates > 0:
                rep_target = teammate_deltas_flat[:, :, 0, :]  # [F, N, D]
            else:
                rep_target = h_i_flat
            h_norm = F.normalize(h_i_flat, dim=-1)
            g_norm = F.normalize(rep_target, dim=-1)
            raw_cos_sim = (h_norm * g_norm).sum(-1).mean().item()

            # NEW (Stage-A-specific): per-(F, i) std across teammate
            # slots.  Directly measures whether different teammates are
            # differently predictable from the same h_i — the *topology-
            # dependent variance* the pairwise fix was meant to unlock.
            # If this is near zero, the predictor outputs similar losses
            # for all teammates → Component 3 of UPMI (identity cond.)
            # may be needed.  If it's large, Stage A is working.
            per_pair_std = per_pair_loss.std(dim=-1)            # [F, N]
            per_pair_std_mean = per_pair_std.mean().item()
            per_pair_std_per_agent = per_pair_std.mean(dim=0).detach()  # [N]

            flat_loss = all_per_agent_loss.reshape(-1)
            q = th.quantile(
                flat_loss, th.tensor([0.1, 0.5, 0.9], device=flat_loss.device)
            )
            loss_p10, loss_p50, loss_p90 = q[0].item(), q[1].item(), q[2].item()

        # Aggregate predictor stats across teammate slots
        top1_stack = th.stack(
            [s["top1_acc_per_agent"] for s in per_teammate_stats], dim=0
        )                                                    # [N-1, N]
        top1_per_agent_avg = top1_stack.mean(dim=0)          # [N]
        pos_score_avg = sum(s["pos_score_mean"] for s in per_teammate_stats) / n_teammates
        neg_score_avg = sum(s["neg_score_mean"] for s in per_teammate_stats) / n_teammates
        margin_avg    = sum(s["margin_mean"]    for s in per_teammate_stats) / n_teammates

        diag = {
            "h_diversity":       h_diversity,
            "delta_diversity":   delta_diversity,
            "delta_norm_mean":   delta_norm_mean,
            # Stage-B-specific (on the ACTUAL predictor signals):
            "local_diversity":       local_diversity,
            "delta_local_diversity": delta_local_diversity,
            "delta_local_norm_mean": delta_local_norm_mean,
            "raw_cos_sim":       raw_cos_sim,
            "loss_p10":          loss_p10,
            "loss_median":       loss_p50,
            "loss_p90":          loss_p90,
            "top1_acc":          top1_per_agent_avg.mean().item(),
            "top1_per_agent":    top1_per_agent_avg,           # [N]
            "pos_score":         pos_score_avg,
            "neg_score":         neg_score_avg,
            "margin":            margin_avg,
            "per_agent_loss_mean":    per_agent_loss.mean(dim=0),  # [N]
            # Stage-A diagnostics:
            "per_pair_std_mean":      per_pair_std_mean,
            "per_pair_std_per_agent": per_pair_std_per_agent,  # [N]
            # Upstream diagnostics (disambiguates predictor vs encoder):
            "h_cross_agent_cos":             h_cross_agent_cos,
            "delta_cross_agent_cos":         delta_cross_agent_cos,
            "local_summary_cross_agent_cos": local_summary_cross_agent_cos,
            "team_summary_cross_agent_cos":  team_summary_cross_agent_cos,
            "delta_local_cross_agent_cos":   delta_local_cross_agent_cos,
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
        # Collect pre- and post-TeamGAT latents too.  These are used
        # ONLY for diagnostics (cross-agent direction cos) so we can
        # pinpoint whether the encoder-level collapse happens before
        # or after TeamGATLayer.  Detached; no gradient path to mac.
        all_local = [] if self.lgdd_enabled else None
        all_team  = [] if self.lgdd_enabled else None
        for t in range(current_batch.max_seq_length):
            if self.lgdd_enabled:
                q_values, latents = self.mac.forward_with_latents(
                    current_batch, t
                )
                all_hidden.append(self.mac.hidden_states.detach().clone())
                all_local.append(latents["local_summary"].detach().clone())
                all_team.append(latents["team_summary"].detach().clone())
            else:
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

            # Component 2: refresh Q-spread EMA from this batch BEFORE
            # computing α (only has effect when mode == q_spread_adaptive).
            if self.alpha_mode == "q_spread_adaptive":
                self._update_q_spread_ema(mac_out, avail_actions, mask)

            # Soft policy from online net (Double-Q: online selects).
            # α is per-agent in general (see alpha_mode); broadcast [N]
            # across [B, T, N, n_actions] by reshaping to [1, 1, N, 1].
            # Curriculum-ramped by t_env (Component 3): early training is
            # uniform, heterogeneity fades in after `alpha_warmup_start`.
            alpha_vec = self._get_alpha_vec(mac_out.device, t_env=t_env)  # [N]
            alpha_bcast = alpha_vec.view(1, 1, -1, 1)                      # [1,1,N,1]

            mac_out_detach = mac_out.clone().detach()
            mac_out_detach = self.mixer.func_f(mac_out_detach, states, t_env)
            mac_out_detach = mac_out_detach / alpha_bcast
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

            # Component 1: decouple sampling-α from target-α.
            # The SAMPLING softmax above uses per-agent α_vec (heterogeneous
            # exploration).  The TARGET entropy bonus below uses the SCALAR
            # α_mean = entropy_coef for every agent.  This makes the
            # regularised objective identical to scalar-α Soft-QMIX (Run A),
            # which removes the per-agent "coordination tax" that caused
            # Run B's late-training stalling attractor.  Heterogeneity is
            # preserved in policy sampling, not in the objective itself.
            target_logp = th.log(actions_pdf + 1e-10)
            target_logp = th.gather(target_logp, 3, picked_actions).squeeze(3)  # [B, T, N]
            alpha_target_scalar = float(self.entropy_coef)
            target_entropy = -alpha_target_scalar * target_logp.sum(-1, keepdim=True)

            # Mix sampled target Q through VDN sum
            target_qvals = self.target_mixer(target_qvals, states)

            # TD(λ) with entropy bonus (α already applied above)
            targets = build_td_lambda_targets(
                rewards, terminated, mask,
                target_qvals, self.n_agents,
                self.args.gamma, self.args.td_lambda,
                target_entropy=target_entropy,
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
                all_hidden, mask, current_batch.device,
                all_local=all_local, all_team=all_team,
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

        # Push the current α vector to the action selector so the next
        # rollout batch uses the same α as this train step.  Cheap — a
        # tiny [n_agents] tensor copy.  Curriculum-ramped via t_env so
        # early-training rollouts also stay on uniform α_mean.
        self.mac.set_alpha(
            self._get_alpha_vec(current_batch.device, t_env=t_env).detach().clone()
        )

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
            # α-mode diagnostics: log the per-agent α used this step so
            # Run B (per_agent_fixed) can be verified and adaptive-mode
            # runs can be debugged.  Curriculum-ramped value.
            alpha_log = self._get_alpha_vec(mac_out.device, t_env=t_env).detach()
            self.logger.log_stat("alpha_mean", alpha_log.mean().item(), t_env)
            self.logger.log_stat("alpha_std",  alpha_log.std().item(),  t_env)
            for i in range(self.n_agents):
                self.logger.log_stat(f"alpha_agent_{i}", alpha_log[i].item(), t_env)

            # Component 3 ramp value (0 early, 1 after warmup_end).
            self.logger.log_stat("alpha_ramp", self._alpha_ramp(t_env), t_env)

            # Component 2 sensor diagnostics (q_spread EMA per-agent).
            # Logged regardless of alpha_mode so scalar / per_agent_fixed
            # runs can still compare the "would-have-been" spread signal.
            if self._q_spread_ema is not None:
                qs = self._q_spread_ema.detach()
                self.logger.log_stat("q_spread_mean", qs.mean().item(), t_env)
                self.logger.log_stat("q_spread_std",  qs.std().item(),  t_env)
                self.logger.log_stat(
                    "q_spread_ratio",
                    (qs.max() / qs.min().clamp(min=1e-6)).item(),
                    t_env,
                )
                for i in range(self.n_agents):
                    self.logger.log_stat(
                        f"q_spread_agent_{i}", qs[i].item(), t_env
                    )
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

                    # ── Formation proxies ───────────────────────────
                    # h_diversity: cross-agent std of absolute hidden states.
                    #              Low = clustered; high = dispersed.
                    # delta_diversity: cross-agent std of Δh_j.  Low = team
                    #                  evolving synchronously (all doing the
                    #                  same thing, including standing still);
                    #                  high = per-agent distinct dynamics.
                    # delta_norm_mean: typical |Δh_j|.  Near zero = stationary
                    #                  episode (bunker/waiting), in which case
                    #                  contrastive comparisons become noisy.
                    self.logger.log_stat(
                        "h_diversity", infonce_diag["h_diversity"], t_env
                    )
                    self.logger.log_stat(
                        "delta_diversity", infonce_diag["delta_diversity"], t_env
                    )
                    self.logger.log_stat(
                        "delta_norm_mean", infonce_diag["delta_norm_mean"], t_env
                    )

                    # ── Stage-A pairwise diagnostics ────────────────
                    # per_pair_std_mean: std across teammate-slot losses,
                    #   averaged over (F, i).  >0 means different teammates
                    #   of the same agent produce different loss values —
                    #   the topology-dependent variance we want Stage A to
                    #   unlock.  Near-zero means all teammates are equally
                    #   predictable from h_i and Stage A's pairwise fix did
                    #   not help (escalate to Stage B or C).
                    self.logger.log_stat(
                        "per_pair_std_mean",
                        infonce_diag["per_pair_std_mean"],
                        t_env,
                    )
                    if "per_pair_std_per_agent" in infonce_diag:
                        for i in range(self.n_agents):
                            self.logger.log_stat(
                                f"per_pair_std_agent_{i}",
                                infonce_diag["per_pair_std_per_agent"][i].item(),
                                t_env,
                            )

                    # ── Upstream (encoder-level) diagnostics ────────
                    # h_cross_agent_cos: average cos(h_i, h_j) for i ≠ j.
                    #   >0.8 → agents' h vectors all point the same way
                    #          → downstream fixes (predictor) won't help
                    #          → need upstream fix (per-agent encoding,
                    #            pre-GAT target, or identity injection).
                    #   <0.5 → agents genuinely differ in direction →
                    #          predictor-collapse is the real issue and
                    #          doubly-conditioned predictor likely helps.
                    # delta_cross_agent_cos: same metric on Δh_j.  If the
                    #   delta operation successfully differentiated agents,
                    #   this should be lower than h_cross_agent_cos.
                    self.logger.log_stat(
                        "h_cross_agent_cos",
                        infonce_diag["h_cross_agent_cos"],
                        t_env,
                    )
                    self.logger.log_stat(
                        "delta_cross_agent_cos",
                        infonce_diag["delta_cross_agent_cos"],
                        t_env,
                    )
                    # ── Encoder-layer-specific cross-agent cos ──────
                    # Pinpoints where the direction-collapse happens:
                    #   local_summary: pre-TeamGAT (agent-local only).
                    #     If <0.5 → TeamGAT is the culprit → Stage B
                    #     (use local_summary as predictor input/target)
                    #     will break the aggregate-uniformity block.
                    #   team_summary: post-TeamGAT.
                    #     If ≈ h_cross_agent_cos → collapse happens at
                    #     or after TeamGAT (no surprise).
                    # delta_local: change in local_summary per step.
                    #     Useful for Stage B target planning.
                    self.logger.log_stat(
                        "local_summary_cross_agent_cos",
                        infonce_diag["local_summary_cross_agent_cos"],
                        t_env,
                    )
                    self.logger.log_stat(
                        "team_summary_cross_agent_cos",
                        infonce_diag["team_summary_cross_agent_cos"],
                        t_env,
                    )
                    self.logger.log_stat(
                        "delta_local_cross_agent_cos",
                        infonce_diag["delta_local_cross_agent_cos"],
                        t_env,
                    )

                    # ── Stage-B signal magnitudes ──────────────────
                    # These describe the ACTUAL signal the predictor
                    # is operating on now (local_summary and Δlocal).
                    # Compare against the legacy h_diversity /
                    # delta_diversity / delta_norm_mean to see what
                    # changed when we switched to pre-TeamGAT.
                    self.logger.log_stat(
                        "local_diversity",
                        infonce_diag["local_diversity"],
                        t_env,
                    )
                    self.logger.log_stat(
                        "delta_local_diversity",
                        infonce_diag["delta_local_diversity"],
                        t_env,
                    )
                    self.logger.log_stat(
                        "delta_local_norm_mean",
                        infonce_diag["delta_local_norm_mean"],
                        t_env,
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
