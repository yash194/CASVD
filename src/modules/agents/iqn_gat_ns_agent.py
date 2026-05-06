"""IQNGATNSAgent: GATNSAgent with a QR-DQN-style K-quantile output head.

This is the distributional Q-head for SYNERGOS Phase 1.  We deliberately
inherit the entire GAT trunk (LocalEntityGAT + TeamGATLayer + GRU) from
GATNSAgent so the only experimental variable in this phase is the head:

    GATNSAgent       :  trunk → GRU → policy_head(hidden) → Q ∈ R^{A}
    IQNGATNSAgent    :  trunk → GRU → policy_head(hidden) → Z ∈ R^{A × K}

The K quantile fractions τ_k = (2k − 1) / (2K) are fixed (QR-DQN style),
so the head is just a wider Linear; the cosine-embedding machinery of the
full IQN (Dabney et al. 2018) is omitted — for fixed β CVaR exploration
the implicit-quantile flexibility is unnecessary, and dropping it makes
the head mathematically equivalent to QR-DQN.

The class name remains "IQN" to preserve future-compat: upgrading to full
IQN means replacing this head with cosine-embedding machinery while keeping
the rest of the pipeline (mixer, selector, learner) unchanged.
"""
import torch as th
import torch.nn as nn

from utils.th_utils import orthogonal_init_

from .gat_ns_agent import GATNSAgent


class IQNGATNSAgent(GATNSAgent):
    def __init__(self, input_shape, args):
        super().__init__(input_shape, args)
        self.K = int(getattr(args, "n_quantiles", 8))

        # Replace the scalar policy head with a K-quantile head.
        # output dim = n_actions * K, reshaped to [B, N, n_actions, K] in forward.
        self.shared_agent.policy_head = nn.Linear(
            self.hidden_dim, self.n_actions * self.K
        )

        # Re-initialise with the same q_head_gain the parent uses, so initial
        # quantile values are near-uniform and exploration is unbiased.
        if getattr(args, "use_orthogonal", False):
            q_head_gain = getattr(args, "q_head_gain", getattr(args, "gain", 1.0))
            orthogonal_init_(self.shared_agent.policy_head, gain=q_head_gain)

    def forward(self, inputs, hidden_state, detach_encoder=False):
        z, next_hidden, _ = self.forward_with_latents(
            inputs, hidden_state, detach_encoder=detach_encoder
        )
        return z, next_hidden

    def forward_with_latents(self, inputs, hidden_state, detach_encoder=False):
        """Forward pass returning the full quantile distribution.

        Returns:
            z:           [B, n_agents, n_actions, K]   raw quantile values (no func_g yet)
            next_hidden: [B, n_agents, hidden_dim]
            latents:     {"local_summary": ..., "team_summary": ...}
        """
        flat_logits, next_hidden, latents = super().forward_with_latents(
            inputs, hidden_state, detach_encoder=detach_encoder
        )
        # super() reshapes to [B, n_agents, *] using -1, so flat_logits is
        # [B, n_agents, n_actions * K].  Split the last dim.
        bs = flat_logits.shape[0]
        z = flat_logits.view(bs, self.n_agents, self.n_actions, self.K)
        return z, next_hidden, latents
