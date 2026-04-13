import torch as th
import torch.nn as nn
import torch.nn.functional as F


class InfoNCEPredictor(nn.Module):
    """InfoNCE coordination sensor for CASVD.

    Measures how well each agent can predict what its TEAMMATES will do next.
    Agent i predicts the mean embedding of all OTHER agents at t+1 (g_others_i),
    NOT the global team mean (which is symmetric and gives identical loss for
    all agents sharing an encoder).

    High InfoNCE loss  → agent can't predict teammates' future → poor coordination
    Low  InfoNCE loss  → agent predicts teammates' future well → good coordination

    This asymmetric target (self excluded from mean) breaks the symmetry of a
    shared encoder, enabling genuine per-agent coord_signal differentiation:
    a unit in the thick of coordinated combat will predict teammate movements
    better than an isolated or dying unit.

    Positive sample: g_others_i[t+1] = mean_{j≠i}(local_summary_j[t+1])
    Negative samples: global team mean at K random timesteps (shared across agents)

    Architecture:
        Two-layer MLP projector [hidden_dim → hidden_dim → hidden_dim].
        Similarity: cosine(W(h_i), g_others) / temperature
        Loss: standard InfoNCE (softmax cross-entropy over positive + K negatives)

    The per-agent loss value, normalised by log(K+1), serves as the
    coordination signal for adaptive alpha_factor in [0, 1].
    """

    def __init__(self, hidden_dim, temperature=0.1):
        super(InfoNCEPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature

        # Two-layer MLP projector: learns nonlinear features that predict global outcomes.
        # Layer 1: hidden_dim → hidden_dim with ReLU (nonlinear extraction)
        # Layer 2: hidden_dim → hidden_dim (projection to similarity space, no bias)
        # Fix 4: replaced single linear W to break coord_signal plateau at ~0.44
        self.W = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
        )
        # Orthogonal init on both linear layers for stable training
        nn.init.orthogonal_(self.W[0].weight)
        nn.init.zeros_(self.W[0].bias)
        nn.init.orthogonal_(self.W[2].weight)

    def forward(self, h_i, g_pos, g_neg):
        """Compute per-agent InfoNCE loss.

        Args:
            h_i:   Agent local embeddings at time t.
                   Shape: [B, n_agents, hidden_dim]
            g_pos: Per-agent positive — mean of OTHER agents' embeddings at t+1.
                   Shape: [B, n_agents, hidden_dim]  (one distinct target per agent)
            g_neg: Shared negatives — global team mean at K random timesteps.
                   Shape: [B, K, hidden_dim]

        Returns:
            loss_per_agent: InfoNCE loss per agent, shape [B, n_agents].
                            Range: [0, log(K+1)].  Lower = better coordination.
        """
        B, n_agents, D = h_i.shape
        K = g_neg.shape[1]

        # Project h_i through W and L2-normalise
        h_proj  = F.normalize(self.W(h_i), dim=-1)   # [B, n_agents, D]
        g_pos_n = F.normalize(g_pos, dim=-1)          # [B, n_agents, D]
        g_neg_n = F.normalize(g_neg, dim=-1)          # [B, K, D]

        # Positive scores: agent i vs its OWN others-future (element-wise dot then sum)
        # h_proj: [B, n_agents, D], g_pos_n: [B, n_agents, D] → [B, n_agents]
        score_pos = (h_proj * g_pos_n).sum(dim=-1) / self.temperature

        # Negative scores: each agent vs K shared random negatives
        # h_proj: [B, n_agents, D], g_neg_n: [B, K, D] → [B, n_agents, K]
        score_neg = th.einsum("bad,bkd->bak", h_proj, g_neg_n) / self.temperature

        # Concatenate: [B, n_agents, 1+K] — positive is index 0
        logits = th.cat([score_pos.unsqueeze(-1), score_neg], dim=-1)

        # Cross-entropy with label=0 (positive is first)
        labels = th.zeros(B, n_agents, dtype=th.long, device=h_i.device)
        loss_per_agent = F.cross_entropy(
            logits.reshape(B * n_agents, 1 + K),
            labels.reshape(B * n_agents),
            reduction="none",
        ).reshape(B, n_agents)

        return loss_per_agent
