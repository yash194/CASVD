"""SyncPredictor — Phase 5b of SYNERGOS.

Multi-timescale predictive synchrony auxiliary.

Inspired by fNIRS/EEG hyperscanning evidence (Reinero et al., 2021;
Kelsen 2026) that successful collaborating teams show *band-specific*
inter-brain synchrony — fast bands carry tactical micro-coordination,
slow bands carry strategic alignment.  The MARL analogue: each agent
predicts other agents' EMA-filtered latents at *both* a fast and a slow
timescale.

Loss.
=====
For each agent i, for each timescale  b ∈ {fast, slow}:

    L_sync,i,b  =  Σ_{j ≠ i}  ‖  ŷ_b(z_i^t)  −  stop_grad( EMA_b(z_j^t) )  ‖²

where  EMA_b(z_j^t)  =  τ_b · EMA_b(z_j^{t-1})  +  (1 − τ_b) · z_j^t  is
the per-agent latent EMA buffer maintained outside this module (the
learner owns the buffers — see synergos_learner.py).  Targets are
detached so the loss only updates ŷ_b and the encoder via the predictor.

Why two bands?
==============
A single timescale either drowns micro coordination in slow noise (high
τ) or fails to track strategy persistence (low τ).  Band 1 (τ_fast =
0.95, ~ 20-step memory) targets immediate tactical alignment; band 2
(τ_slow = 0.99, ~ 100-step memory) targets strategy persistence across
combat phases.  Two-band loss is published-validated and computationally
trivial (O(N²·D) per step).
"""
import torch as th
import torch.nn as nn


class SyncPredictor(nn.Module):
    def __init__(self, hidden_dim, n_bands=2, n_agents=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_bands = n_bands
        self.n_agents = n_agents

        in_dim = hidden_dim + (n_agents if n_agents is not None else 0)
        # One MLP head per band.  Predict the teammate's EMA-filtered latent
        # at the matching timescale.
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(n_bands)
        ])
        for head in self.heads:
            for layer in head:
                if isinstance(layer, nn.Linear):
                    nn.init.orthogonal_(layer.weight, gain=1.0)
                    nn.init.zeros_(layer.bias)

    def forward(self, z_now, z_emas):
        """
        z_now:    [B, T, N, D]   per-agent latents at current timestep
        z_emas:   list of length n_bands, each [B, T, N, D]
                  EMA-filtered latents (detached) per band.

        Returns scalar synchrony loss (mean over B × T × bands × cross-agent pairs).
        """
        B, T, N, D = z_now.shape
        assert len(z_emas) == self.n_bands

        if self.n_agents is not None:
            id_oh = th.eye(N, device=z_now.device).view(1, 1, N, N).expand(B, T, -1, -1)
            z_in = th.cat([z_now, id_oh], dim=-1)
        else:
            z_in = z_now

        total_loss = z_now.new_zeros(())
        # Cross-agent indices: all (i, j) with i ≠ j, computed once.
        idx = th.arange(N, device=z_now.device)
        i_idx, j_idx = th.meshgrid(idx, idx, indexing="ij")
        mask = (i_idx != j_idx)                                             # [N, N]
        i_keep = i_idx[mask]                                                # [N(N-1)]
        j_keep = j_idx[mask]

        for b in range(self.n_bands):
            ema_b = z_emas[b].detach()                                      # [B, T, N, D]
            pred_b = self.heads[b](z_in)                                    # [B, T, N, D]
            # For each (i, j ≠ i): squared error between pred_i and ema_j
            pred_pair = pred_b[:, :, i_keep, :]                             # [B, T, N(N-1), D]
            target_pair = ema_b[:, :, j_keep, :]
            sq = (pred_pair - target_pair).pow(2).mean(dim=-1)              # [B, T, N(N-1)]
            total_loss = total_loss + sq.mean()

        return total_loss / max(self.n_bands, 1)
