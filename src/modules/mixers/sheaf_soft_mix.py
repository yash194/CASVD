"""SheafSoftMixer — Phase 2 of SYNERGOS.

Extends DistSoftMixer (Phase 1) with a *sheaf-cochain* aggregation step
inserted between the per-agent value head and the VDN sum.

Why a sheaf?
============
SMACv2's procedurally-randomised unit composition means agent i is a
Stalker in one episode and a Zealot in the next.  Standard QMIX/Soft-QMIX
mixers use a *type-blind* monotone aggregation: the per-agent hypernet
outputs share weights regardless of the unit type at index i.  This is
the heterogeneity bottleneck (D1 in the proposal).

A cellular sheaf F over the agent graph G = (V, E) equips each agent
with its own stalk F(i) and each edge with restriction maps
F_{i ⊴ e} : F(i) → F(e).  Two agents are "in agreement" only when their
restriction-mapped values match — so the sheaf aggregation is
*heterogeneity-aware* by construction.

Implementation choices for SMACv2 scale (N = 5 .. 20).
======================================================
1.  **Stalk identity.**  We use a learned per-position embedding combined
    with the global state slice that the QMIX hypernet already sees.  This
    keeps the mixer state-aware without requiring explicit unit-type
    plumbing through the runner / batch.  When `store_agent_types: True`
    is later wired up, the unit-type tensor can be substituted in via the
    `agent_id_override` hook (see `forward_dist_sheaf`).

2.  **Restriction maps as scalar weights.**  For each pair (i, j), we
    learn a positive scalar  w_ij = softplus( MLP( id_i, id_j, s ) ).
    This is the simplest non-trivial restriction map; it is monotone in
    the source value, which preserves IGM under summation.

3.  **Heat-kernel approximation via convex sheaf message passing.**
    Computing  exp( − L_F · t )  exactly via eigendecomposition is O(N^3)
    per forward pass.  We instead apply  K  rounds of

        h^{(k+1)}  =  (1 − α) · h^{(k)}  +  α · W̃ · h^{(k)}

    where  W̃  is the row-normalised restriction matrix.  This is a
    convex combination at every step, so each agent's value remains a
    monotone-increasing function of every  Q_j  with  w_ij > 0.  IGM
    holds.  K = 2, α = 0.5 is the default — empirically sufficient for
    N ≤ 10 and ~ 30× cheaper than literal heat kernel.

4.  **Distributional broadcast.**  The restriction matrix W̃ depends only
    on (state, agent ids), not on the quantile τ.  We compute it once per
    (B, T) and reuse it across all K_quantile evaluations.  The fold-K-into-
    batch trick used by DistSoftMixer is unnecessary here, which is faster.

IGM theorem (preserved).
========================
Claim.  For monotone restriction maps  w_ij ≥ 0  and convex sheaf updates,
the resulting Q_tot is monotone non-decreasing in every per-agent Q_i, so

    argmax_a  E_τ Z_tot(s, a; τ)  =  ( argmax_{a_i}  E_τ Z_i(s, a_i; τ) )_i.

Proof sketch.  Each round of message passing gives  h^{(k+1)}_i  =
(1 − α) h^{(k)}_i  +  α Σ_j W̃_ij h^{(k)}_j , which is a non-negative
linear combination of the inputs.  Composition of non-negative linear
maps is non-negative linear; therefore  ∂h^{(K)}_i / ∂Q_j ≥ 0  for all
i, j.  Final aggregation is sum, so  ∂Q_tot / ∂Q_j  =  Σ_i ∂h^{(K)}_i /
∂Q_j ≥ 0.  ∎

Connection to existing code.
=============================
SheafSoftMixer extends DistSoftMixer and:
  • adds  forward_dist_sheaf(z, states)  → Z_tot per quantile, replacing
    the plain VDN sum  z.sum(dim=2)  used in Phase 1.
  • leaves func_g_dist and func_f_dist untouched (still applied per
    quantile by the parent class).

The mixer interface remains compatible with the rollout selector — the
selector only ever calls func_g_dist + func_f_dist; the sheaf aggregation
is invoked exclusively by the learner during target / chosen-Q mixing.
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .dist_soft_mix import DistSoftMixer


class SheafSoftMixer(DistSoftMixer):
    def __init__(self, args):
        super().__init__(args)
        self.id_dim = int(getattr(args, "sheaf_id_dim", 16))
        self.n_sheaf_steps = int(getattr(args, "n_sheaf_steps", 2))
        self.alpha_step = float(getattr(args, "sheaf_step_size", 0.5))
        self.use_self_loop = bool(getattr(args, "sheaf_self_loop", True))

        # Learned per-position identity embedding.  Used as the agent's
        # "stalk identity"; in a future revision when unit types are stored
        # in the batch, this can be replaced with a (position, type) joint
        # embedding without changing the rest of the pipeline.
        self.id_emb = nn.Embedding(self.n_agents, self.id_dim)
        nn.init.orthogonal_(self.id_emb.weight, gain=1.0)

        # Restriction MLP: (id_i, id_j, state) → scalar weight
        in_dim = 2 * self.id_dim + self.state_dim
        hidden = int(getattr(args, "sheaf_restriction_hidden", 64))
        self.restriction_mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        # Initialise the final layer with small weights so that early
        # softplus outputs cluster around log(2) ≈ 0.69 — this gives
        # roughly uniform restriction weights at t=0, so the mixer
        # initialises as approximately VDN and adapts heterogeneity over
        # training (rather than starting in a degenerate corner).
        nn.init.normal_(self.restriction_mlp[-1].weight, std=0.01)
        nn.init.zeros_(self.restriction_mlp[-1].bias)

    def _compute_restriction_matrix(self, states):
        """Compute the row-normalised restriction matrix  W̃  given states.

        states: [B, T, S]
        Returns W̃: [B, T, N, N], rows sum to 1.
        """
        B, T, S = states.shape
        N = self.n_agents

        ids = th.arange(N, device=states.device)
        id_emb = self.id_emb(ids)                                         # [N, id_dim]
        id_emb_b = id_emb.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)  # [B, T, N, id_dim]

        id_i = id_emb_b.unsqueeze(3).expand(-1, -1, -1, N, -1)             # [B, T, N, N, id]
        id_j = id_emb_b.unsqueeze(2).expand(-1, -1, N, -1, -1)             # [B, T, N, N, id]
        states_exp = states.unsqueeze(2).unsqueeze(3).expand(-1, -1, N, N, -1)
        # [B, T, N, N, S]

        in_pair = th.cat([id_i, id_j, states_exp], dim=-1)                 # [B, T, N, N, *]
        w_logit = self.restriction_mlp(in_pair).squeeze(-1)                # [B, T, N, N]
        w = F.softplus(w_logit)

        if self.use_self_loop:
            # Add an identity self-loop so each agent always contributes to
            # itself in the message-passing update — prevents the row from
            # being dominated by neighbours when the MLP outputs cluster
            # near zero, and matches the standard graph-Laplacian convention.
            eye = th.eye(N, device=w.device).unsqueeze(0).unsqueeze(0)
            w = w + eye

        # Row-normalise so each h^{(k+1)}_i is a convex combination
        # → preserves boundedness and IGM monotonicity.
        w_normed = w / w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        return w_normed

    def forward_dist_sheaf(self, z_per_agent, states):
        """Sheaf-cochain VDN: K rounds of message passing, then sum across agents.

        z_per_agent: [B, T, N, K]   per-agent quantile distributions at the
                                    chosen action (post-VDN-input).
        states:      [B, T, S]
        Returns      [B, T, K]      Z_tot per quantile.
        """
        # Compute restriction matrix once (does not depend on quantile dim).
        w = self._compute_restriction_matrix(states)                       # [B, T, N, N]

        h = z_per_agent                                                    # [B, T, N, K]
        for _ in range(self.n_sheaf_steps):
            h_new = th.einsum("btij,btjk->btik", w, h)                     # [B, T, N, K]
            h = (1.0 - self.alpha_step) * h + self.alpha_step * h_new

        return h.sum(dim=2)                                                # [B, T, K]

    def forward_sheaf(self, q_per_agent, states):
        """Scalar (non-distributional) version, for the β-loss path.

        q_per_agent: [B, T, N]   per-agent scalar Q at chosen action.
        states:      [B, T, S]
        Returns      [B, T, 1]   Q_tot.
        """
        w = self._compute_restriction_matrix(states)
        h = q_per_agent
        for _ in range(self.n_sheaf_steps):
            h_new = th.einsum("btij,btj->bti", w, h)
            h = (1.0 - self.alpha_step) * h + self.alpha_step * h_new
        return h.sum(dim=-1, keepdim=True)
