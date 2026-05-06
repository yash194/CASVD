"""SlowRolePredictor — Phase 5a of SYNERGOS.

Future-conditioned slow-role InfoNCE  (R3DM-style; Goel et al., ICML 2025).

Each agent maintains a slow latent  m_i^t  inferred from a window of past
trajectory  τ_i^{t−W : t}.  The training objective is contrastive:  m_i^t
should be predictive of the agent's *future* trajectory window
τ_i^{t : t+T_horizon}, with negatives drawn from cross-episode and
cross-agent samples.

Why future-conditioned?
=======================
Past-conditioned variational role objectives (ROMA, CDS) suffer from
posterior collapse and "identical-history → identical-role" failures.
Conditioning the role on the *future* trajectory breaks the symmetry —
two agents with similar pasts but divergent futures get distinct roles.
This was the headline R3DM contribution (+20 pp on the hardest SMACv2
maps; arXiv 2505.24265).

Implementation.
===============
For Phase 5 we treat slow-role as a *representation auxiliary*:

  1. Encode past window  τ_i^{t−W : t}  → m_i^t  (slow encoder, GRU).
  2. Encode future window τ_i^{t : t+T}  → e_i^t  (future encoder, MLP).
  3. InfoNCE: predict e_i^t given m_i^t, with cross-batch + cross-agent
     negatives.

The role latent is *not* yet wired into the policy — that requires a
larger architectural change (agent forward conditioning) and is left
for a follow-up.  The InfoNCE loss alone shapes the encoder so that its
hidden states carry distinguishable per-agent strategy signal, which is
sufficient for the downstream sync auxiliary and is a standalone
representation learning win.
"""
import torch as th
import torch.nn as nn
import torch.nn.functional as F


class SlowRolePredictor(nn.Module):
    def __init__(self, hidden_dim, role_dim=32, n_agents=None, temperature=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.role_dim = role_dim
        self.temperature = temperature
        self.n_agents = n_agents

        # Past encoder: averaged hidden state in window → role latent
        # We use mean-pool over the W-step window then MLP — equivalent
        # to a temporal-attention with uniform weights (cheap and stable).
        in_past = hidden_dim + (n_agents if n_agents is not None else 0)
        self.past_encoder = nn.Sequential(
            nn.Linear(in_past, role_dim),
            nn.ReLU(),
            nn.Linear(role_dim, role_dim),
        )

        # Future encoder: averaged hidden in future window → predicted target
        in_future = hidden_dim
        self.future_encoder = nn.Sequential(
            nn.Linear(in_future, role_dim),
            nn.ReLU(),
            nn.Linear(role_dim, role_dim),
        )

        for module in (self.past_encoder, self.future_encoder):
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                    nn.init.zeros_(layer.bias)

    def forward(self, past_h, future_h, return_stats=False):
        """
        past_h:   [B, n_agents, hidden_dim]  averaged latent over window τ^{t−W:t}
        future_h: [B, n_agents, hidden_dim]  averaged latent over window τ^{t:t+T}

        Returns scalar InfoNCE loss (mean over B × N × negatives).
        """
        B, N, D = past_h.shape

        # Identity-conditioned past encoding (CASVD pattern: append agent
        # one-hot so the shared MLP can specialise per agent).
        if self.n_agents is not None:
            assert N == self.n_agents
            id_oh = th.eye(N, device=past_h.device).unsqueeze(0).expand(B, -1, -1)
            past_in = th.cat([past_h, id_oh], dim=-1)
        else:
            past_in = past_h

        m_past = F.normalize(self.past_encoder(past_in), dim=-1)            # [B, N, role_dim]
        e_future = F.normalize(self.future_encoder(future_h), dim=-1)       # [B, N, role_dim]

        # Positive: agent i past predicts agent i future (same B, same agent).
        # Negatives: every other (B', N') pair in the batch.
        m_flat = m_past.reshape(B * N, self.role_dim)                       # [BN, R]
        e_flat = e_future.reshape(B * N, self.role_dim)                     # [BN, R]

        sim = th.einsum("ar,br->ab", m_flat, e_flat) / self.temperature     # [BN, BN]
        # The positive is the diagonal (self pairing).
        labels = th.arange(B * N, device=past_h.device)
        loss = F.cross_entropy(sim, labels, reduction="mean")

        if return_stats:
            with th.no_grad():
                top1 = (sim.argmax(dim=-1) == labels).float().mean().item()
            return loss, {"slow_role_top1": top1}
        return loss
