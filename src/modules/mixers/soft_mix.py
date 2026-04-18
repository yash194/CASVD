import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SoftMixer(nn.Module):
    """Soft-QMIX mixer: VDN forward + func_f (affine) + func_g (order-preserving).

    Forward pass is plain VDN (sum of per-agent Q-values).  State
    conditioning enters through two monotonic transformations:

    func_g  — nonlinear order-preserving residual applied to Q-values
              before action selection and target computation.
    func_f  — affine per-agent transformation applied before the soft
              policy.  Acts as learned per-agent, per-state temperature.

    Both use positive weights to guarantee IGM.
    """

    def __init__(self, args):
        super(SoftMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.state_dim = int(np.prod(args.state_shape))
        hypernet_embed = getattr(args, "hypernet_embed", 64)

        # ── func_g: order-preserving residual ─────────────────────
        # y = ELU(w3 * Q + b3) * w4 + b4 + Q   (w3, w4 > 0)
        self.hyper_w3 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(hypernet_embed, self.n_agents * self.embed_dim // 2),
        )
        self.hyper_b3 = nn.Sequential(
            nn.Linear(self.state_dim, self.n_agents * self.embed_dim // 2),
        )
        self.hyper_w4 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(hypernet_embed, self.n_agents * self.embed_dim // 2),
        )
        self.hyper_b4 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(hypernet_embed, self.n_agents),
        )

        # ── func_f: affine per-agent transformation ──────────────
        # y = w5 * Q + b5   (w5 > 0)
        self.hyper_w5 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(hypernet_embed, self.n_agents),
        )
        self.hyper_b5 = nn.Sequential(
            nn.Linear(self.state_dim, hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(hypernet_embed, self.n_agents),
        )

    def func_f(self, qvals, states, t_env=None, death_mask=None):
        """Affine: w5(s) * Q_i + b5(s), w5 > 0."""
        qval_shape = qvals.shape
        states_flat = states.reshape(-1, self.state_dim)

        if qval_shape[-2] == self.n_agents:
            qvals_r = qvals.reshape(-1, self.n_agents, qvals.shape[-1])
            w = self.hyper_w5(states_flat).view(-1, self.n_agents, 1)
            b = self.hyper_b5(states_flat).view(-1, self.n_agents, 1)
        elif qval_shape[-1] == self.n_agents:
            qvals_r = qvals.reshape(-1, self.n_agents)
            w = self.hyper_w5(states_flat).view(-1, self.n_agents)
            b = self.hyper_b5(states_flat).view(-1, self.n_agents)
        else:
            raise ValueError(
                f"func_f: unexpected qval shape {qval_shape} for n_agents={self.n_agents}"
            )

        w = w.abs()
        return (qvals_r * w + b).reshape(qval_shape)

    def func_g(self, qvals, states, t_env=None, death_mask=None):
        """Order-preserving residual: ELU(w3*Q+b3)*w4 + b4 + Q."""
        qval_shape = qvals.shape
        states_flat = states.reshape(-1, self.state_dim)

        qvals_r = qvals.reshape(-1, 1, self.n_agents, qval_shape[-1])
        w1 = self.hyper_w3(states_flat).view(-1, self.embed_dim // 2, self.n_agents, 1)
        b1 = self.hyper_b3(states_flat).view(-1, self.embed_dim // 2, self.n_agents, 1)
        w2 = self.hyper_w4(states_flat).view(-1, self.embed_dim // 2, self.n_agents, 1)
        b2 = self.hyper_b4(states_flat).view(-1, 1, self.n_agents, 1)

        w1 = w1.abs()
        w2 = w2.abs()

        y = F.elu(qvals_r * w1 + b1)
        y = (y * w2).sum(dim=-3, keepdim=True) + b2
        y = y + qvals_r
        return y.reshape(qval_shape)

    def forward(self, qvals, states, death_mask=None):
        """VDN: Q_tot = sum of per-agent Q-values."""
        b, t, _ = qvals.size()
        return qvals.sum(-1).view(b, t, -1)
