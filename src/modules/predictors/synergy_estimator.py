"""SynergyEstimator — Phase 4 of SYNERGOS.

Estimates the *synergistic* atom of the Williams-Beer Partial Information
Decomposition for cooperative multi-agent return:

    I(a_1, ..., a_N ; G | s)  =  R  +  Σ_i U_i  +  S

where R is redundant info (reachable from any single agent), U_i is unique
info (only agent i has it), and S is **synergistic info** — joint-only
information about the team return G.

Standard cooperative-MARL value decomposition rewards R + U adequately but
implicitly *underweights* S.  A team that is "more than the sum of its
parts" is precisely a team whose actions carry high S relative to the
joint mutual information.  SYNERGOS Phase 4 adds an explicit synergy
intrinsic reward that incentivises joint-only behaviours.

Estimator (CCS lower bound, Ince 2017).
=======================================
For each agent pair (i, j) we estimate two contrastive lower-bound MIs:

    I_pair_{ij}  ≥  E[ log f_pair(s, a_i, a_j, G_pos)
                       − log Σ_neg exp f_pair(s, a_i, a_j, G_neg) ]

    I_single_{k} ≥  E[ log f_single(s, a_k, G_pos)
                       − log Σ_neg exp f_single(s, a_k, G_neg) ]

The synergy lower bound is

    S̃_{ij}  =  max( 0,  I_pair_{ij}  −  max_k I_single_{k} ).

Per-agent synergy attribution:

    S̃_i^t  =  (1 / (N − 1))  Σ_{j ≠ i}  S̃_{ij}^t.

Per-agent reward shaping:

    r̃_i^t  =  r^t  +  η · S̃_i^t,                          η ≥ 0,

with the safety bound  η · max S̃ < (1 − γ) · α · log|A|  enforced by the
caller (clip in the learner).  Under this bound, the synergy bonus acts
as a potential-shaped reward and does not change the optimal policy
(see proposal §5.3).

Implementation choices.
=======================
1.  **Cross-batch negatives.**  G_pos for sample t is the n-step return at
    that timestep; G_neg is sampled from other (B, T) positions in the
    same batch.  Cheap, decorrelated, well-conditioned.
2.  **Single discriminator per role.**  We share one f_pair and one
    f_single across all (i, j); positions are encoded by appending an
    index one-hot to the input.  This avoids O(N) parameter blow-up and
    matches CASVD's identity-conditioned predictor pattern.
3.  **Gradient isolation.**  Discriminator parameters live in their own
    optimiser; the synergy bonus is `.detach()`-ed before being added to
    the per-step reward, so no gradient flows from the RL loss into the
    discriminator and vice versa.
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class SynergyEstimator(nn.Module):
    def __init__(self, state_dim, n_actions, n_agents, hidden=64, temperature=0.5):
        super().__init__()
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.n_agents = n_agents
        self.temperature = temperature

        # Pair scorer: f_pair(s, a_i, a_j, idx_i, idx_j, G) -> R
        # Encoded inputs:
        #   s          : [..., S]
        #   a_i, a_j   : [..., A]    one-hot
        #   idx_i, j   : [..., N]    one-hot agent-index
        #   G          : [..., 1]    scalar return
        in_pair = state_dim + 2 * n_actions + 2 * n_agents + 1
        self.f_pair = nn.Sequential(
            nn.Linear(in_pair, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        # Single scorer: f_single(s, a_k, idx_k, G) -> R
        in_single = state_dim + n_actions + n_agents + 1
        self.f_single = nn.Sequential(
            nn.Linear(in_single, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

        for module in (self.f_pair, self.f_single):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                    nn.init.zeros_(layer.bias)

    @staticmethod
    def _onehot(idx, depth):
        """Long [..., 1] → float [..., depth]."""
        flat = idx.reshape(-1, 1)
        out = th.zeros(flat.shape[0], depth, device=idx.device, dtype=th.float32)
        out.scatter_(1, flat.long(), 1.0)
        return out.reshape(*idx.shape[:-1], depth)

    def compute(self, states_flat, actions_onehot_flat, G_flat,
                idx_i_flat, idx_j_flat, idx_k_flat,
                n_negatives=15):
        """Compute per-sample synergy lower bounds and InfoNCE loss.

        All inputs are flattened to [F, *].  F is the number of valid samples
        in the batch (B*T-1 after masking).

        Args:
            states_flat:        [F, S]
            actions_onehot_flat:[F, N, A]
            G_flat:             [F, 1]
            idx_i_flat:         [F, 1] — agent index for "first slot of pair"
            idx_j_flat:         [F, 1] — agent index for "second slot of pair"
            idx_k_flat:         [F, 1] — agent index for "single comparison"
            n_negatives:        int — number of cross-batch G negatives

        Returns:
            synergy_per_sample: [F, 1]  S̃_ij ≥ 0  for the supplied (i, j, k) triple
            loss:               scalar — InfoNCE training loss for the discriminators.
        """
        F_dim = states_flat.shape[0]
        device = states_flat.device

        # Sample negatives (G drawn from random other positions in the batch).
        if F_dim < 2:
            # Degenerate batch: synergy is zero, no meaningful loss.
            return th.zeros(F_dim, 1, device=device), th.zeros((), device=device)
        neg_idx = th.randint(0, F_dim, (F_dim, n_negatives), device=device)
        # Ensure no positive collisions (cheap rejection: shift any equal-to-self).
        self_idx = th.arange(F_dim, device=device).unsqueeze(-1)
        coll = (neg_idx == self_idx)
        neg_idx = th.where(coll, (neg_idx + 1) % F_dim, neg_idx)

        G_neg = G_flat[neg_idx]                                            # [F, K, 1]

        # Encode action lookups.
        a_i = self._gather_action(actions_onehot_flat, idx_i_flat)         # [F, A]
        a_j = self._gather_action(actions_onehot_flat, idx_j_flat)
        a_k = self._gather_action(actions_onehot_flat, idx_k_flat)
        oh_i = self._onehot(idx_i_flat, self.n_agents)                     # [F, N]
        oh_j = self._onehot(idx_j_flat, self.n_agents)
        oh_k = self._onehot(idx_k_flat, self.n_agents)

        # ── pair scorer ───────────────────────────────────────────────
        s_pair_pos = th.cat([states_flat, a_i, a_j, oh_i, oh_j, G_flat], dim=-1)
        score_pair_pos = self.f_pair(s_pair_pos).squeeze(-1) / self.temperature  # [F]

        K = n_negatives
        s_pair_neg = th.cat([
            states_flat.unsqueeze(1).expand(-1, K, -1),
            a_i.unsqueeze(1).expand(-1, K, -1),
            a_j.unsqueeze(1).expand(-1, K, -1),
            oh_i.unsqueeze(1).expand(-1, K, -1),
            oh_j.unsqueeze(1).expand(-1, K, -1),
            G_neg,
        ], dim=-1)                                                          # [F, K, *]
        score_pair_neg = self.f_pair(s_pair_neg).squeeze(-1) / self.temperature  # [F, K]

        logits_pair = th.cat([score_pair_pos.unsqueeze(-1), score_pair_neg], dim=-1)
        labels = th.zeros(F_dim, dtype=th.long, device=device)
        loss_pair = F.cross_entropy(logits_pair, labels, reduction="mean")

        # Lower bound  I_pair  ≥  log(K+1) − loss_pair  (NCE bound).
        log_K1 = th.log(th.tensor(float(K + 1), device=device))
        I_pair = (log_K1 - loss_pair).clamp(min=0.0)                        # scalar

        # ── single scorer ─────────────────────────────────────────────
        s_single_pos = th.cat([states_flat, a_k, oh_k, G_flat], dim=-1)
        score_single_pos = self.f_single(s_single_pos).squeeze(-1) / self.temperature

        s_single_neg = th.cat([
            states_flat.unsqueeze(1).expand(-1, K, -1),
            a_k.unsqueeze(1).expand(-1, K, -1),
            oh_k.unsqueeze(1).expand(-1, K, -1),
            G_neg,
        ], dim=-1)
        score_single_neg = self.f_single(s_single_neg).squeeze(-1) / self.temperature

        logits_single = th.cat([score_single_pos.unsqueeze(-1), score_single_neg], dim=-1)
        loss_single = F.cross_entropy(logits_single, labels, reduction="mean")
        I_single = (log_K1 - loss_single).clamp(min=0.0)

        # Synergy lower bound (max with redundancy approximated by I_single
        # at the queried agent k — see proposal §3.1).
        S_lb = (I_pair - I_single).clamp(min=0.0)
        synergy_per_sample = S_lb.expand(F_dim, 1)                          # broadcast

        # Combined discriminator training loss.
        loss = 0.5 * (loss_pair + loss_single)
        return synergy_per_sample, loss

    @staticmethod
    def _gather_action(actions_onehot_flat, idx_flat):
        """actions_onehot_flat: [F, N, A], idx_flat: [F, 1] long → [F, A]."""
        F_dim, N, A = actions_onehot_flat.shape
        idx_exp = idx_flat.long().view(F_dim, 1, 1).expand(-1, 1, A)
        return th.gather(actions_onehot_flat, dim=1, index=idx_exp).squeeze(1)
