"""DistSoftMixer: SoftMixer (Soft-QMIX VDN + func_g + func_f) lifted to a
quantile-indexed Q-distribution.

Used by SYNERGOS Phase 1 (Distributional Soft-QMIX with CVaR exploration).

Tensor convention.
    Distributional Q with action dim:        Z ∈ R^{B × T × N × A × K}
    Distributional Q at the chosen action:   Z ∈ R^{B × T × N × K}

The quantile dim K is always last.  Each method folds K into the leading
batch dim, calls the parent's func_f / func_g (which are already monotone
elementwise per agent and per action), and unfolds.  The state hypernets
do not see K, so the per-quantile transformations share weights — this is
what preserves IGM in the distributional case.
"""
import torch as th

from .soft_mix import SoftMixer


class DistSoftMixer(SoftMixer):
    def func_g_dist(self, z, states, t_env=None):
        """Apply func_g per quantile.

        z:      [B, T, N, A, K]
        states: [B, T, S]
        Returns [B, T, N, A, K].
        """
        B, T, N, A, K = z.shape
        # Move K to leading: [B, K, T, N, A]
        z_perm = z.permute(0, 4, 1, 2, 3).contiguous()
        z_flat = z_perm.reshape(B * K, T, N, A)
        # Broadcast states across K
        states_exp = states.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, T, -1)
        out_flat = self.func_g(z_flat, states_exp, t_env)
        # Unfold: [B*K, T, N, A] -> [B, K, T, N, A] -> [B, T, N, A, K]
        out = out_flat.reshape(B, K, T, N, A).permute(0, 2, 3, 4, 1).contiguous()
        return out

    def func_f_dist(self, z, states, t_env=None):
        """Apply func_f per quantile.

        Supports two shapes:
          z [B, T, N, A, K]  → distributional Q before action selection
          z [B, T, N, K]     → distributional Q at the chosen action
        """
        if z.dim() == 5:
            B, T, N, A, K = z.shape
            z_perm = z.permute(0, 4, 1, 2, 3).contiguous()
            z_flat = z_perm.reshape(B * K, T, N, A)
            states_exp = states.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, T, -1)
            out_flat = self.func_f(z_flat, states_exp, t_env)
            out = out_flat.reshape(B, K, T, N, A).permute(0, 2, 3, 4, 1).contiguous()
            return out
        elif z.dim() == 4:
            B, T, N, K = z.shape
            z_perm = z.permute(0, 3, 1, 2).contiguous()
            z_flat = z_perm.reshape(B * K, T, N)
            states_exp = states.unsqueeze(1).expand(-1, K, -1, -1).reshape(B * K, T, -1)
            out_flat = self.func_f(z_flat, states_exp, t_env)
            out = out_flat.reshape(B, K, T, N).permute(0, 2, 3, 1).contiguous()
            return out
        else:
            raise ValueError(
                f"func_f_dist: expected z of dim 4 or 5, got {z.dim()} (shape {tuple(z.shape)})"
            )

    def forward_dist(self, z_per_agent, states):
        """VDN sum across agents per quantile.

        z_per_agent: [B, T, N, K]
        Returns      [B, T, K].
        """
        return z_per_agent.sum(dim=2)
