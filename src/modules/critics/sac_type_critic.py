import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from utils.th_utils import orthogonal_init_


class TwinQNetwork(nn.Module):
    def __init__(self, input_shape, hidden_dim, n_actions, use_layer_norm=False):
        super(TwinQNetwork, self).__init__()
        self.use_layer_norm = use_layer_norm
        self.q1_fc1 = nn.Linear(input_shape, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, n_actions)
        if self.use_layer_norm:
            self.q1_ln = LayerNorm(hidden_dim)

        self.q2_fc1 = nn.Linear(input_shape, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, n_actions)
        if self.use_layer_norm:
            self.q2_ln = LayerNorm(hidden_dim)

    def forward(self, x):
        q1 = F.relu(self.q1_fc1(x), inplace=True)
        q1 = F.relu(self.q1_fc2(q1), inplace=True)
        if self.use_layer_norm:
            q1 = self.q1_ln(q1)
        q1 = self.q1_out(q1)

        q2 = F.relu(self.q2_fc1(x), inplace=True)
        q2 = F.relu(self.q2_fc2(q2), inplace=True)
        if self.use_layer_norm:
            q2 = self.q2_ln(q2)
        q2 = self.q2_out(q2)
        return q1, q2


class GlobalCoordinator(nn.Module):
    def __init__(self, n_types, n_actions, hidden_dim):
        super(GlobalCoordinator, self).__init__()
        self.fc1 = nn.Linear(n_types * n_actions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def forward(self, type_qs):
        return self.fc2(F.relu(self.fc1(type_qs), inplace=True))


class SACTypeCritic(nn.Module):
    def __init__(self, scheme, args):
        super(SACTypeCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.hidden_dim = args.hidden_dim

        self.unit_types = list(getattr(args, "unit_types", ["default"]))
        if len(self.unit_types) == 0:
            self.unit_types = ["default"]
        self.n_types = len(self.unit_types)
        self.use_global = self.n_types >= 2
        self.use_layer_norm = getattr(args, "use_layer_norm", False)

        input_shape = self._get_input_shape(scheme)
        self.type_critics = nn.ModuleList(
            [
                TwinQNetwork(
                    input_shape,
                    self.hidden_dim,
                    self.n_actions,
                    use_layer_norm=self.use_layer_norm,
                )
                for _ in range(self.n_types)
            ]
        )
        if self.use_global:
            self.global_coord_q1 = GlobalCoordinator(self.n_types, self.n_actions, self.hidden_dim)
            self.global_coord_q2 = GlobalCoordinator(self.n_types, self.n_actions, self.hidden_dim)

        if getattr(args, "use_orthogonal", False):
            gain = getattr(args, "gain", 1.0)
            for type_critic in self.type_critics:
                for module in [
                    type_critic.q1_fc1,
                    type_critic.q1_fc2,
                    type_critic.q1_out,
                    type_critic.q2_fc1,
                    type_critic.q2_fc2,
                    type_critic.q2_out,
                ]:
                    orthogonal_init_(module, gain=gain)
            if self.use_global:
                for module in [
                    self.global_coord_q1.fc1,
                    self.global_coord_q1.fc2,
                    self.global_coord_q2.fc1,
                    self.global_coord_q2.fc2,
                ]:
                    orthogonal_init_(module, gain=gain)

    def forward(self, batch, agent_type_ids=None, t=None):
        inputs, bs, max_t = self._build_inputs(batch, t=t)

        if agent_type_ids is None:
            if "agent_types" in batch.scheme:
                agent_type_ids = batch["agent_types"].long().to(inputs.device)
            else:
                agent_type_ids = th.zeros(bs, self.n_agents, dtype=th.long, device=inputs.device)

        if not self.use_global:
            return self.type_critics[0](inputs)

        q1_out = th.zeros(bs, max_t, self.n_agents, self.n_actions, device=inputs.device)
        q2_out = th.zeros_like(q1_out)

        all_q1s = []
        all_q2s = []
        for critic in self.type_critics:
            q1_t, q2_t = critic(inputs)
            all_q1s.append(q1_t)
            all_q2s.append(q2_t)

        for agent_idx in range(self.n_agents):
            type_idx = agent_type_ids[:, agent_idx].clamp(min=0, max=self.n_types - 1)
            agent_q1_all = th.stack([q1[:, :, agent_idx, :] for q1 in all_q1s], dim=2)
            agent_q2_all = th.stack([q2[:, :, agent_idx, :] for q2 in all_q2s], dim=2)

            q1_flat = agent_q1_all.reshape(bs * max_t, self.n_types * self.n_actions)
            q2_flat = agent_q2_all.reshape(bs * max_t, self.n_types * self.n_actions)
            q1_coord = self.global_coord_q1(q1_flat).view(bs, max_t, self.n_actions)
            q2_coord = self.global_coord_q2(q2_flat).view(bs, max_t, self.n_actions)

            gather_idx = type_idx.view(bs, 1, 1, 1).expand(-1, max_t, 1, self.n_actions)
            q1_type = agent_q1_all.gather(2, gather_idx).squeeze(2)
            q2_type = agent_q2_all.gather(2, gather_idx).squeeze(2)
            q1_out[:, :, agent_idx, :] = q1_coord + q1_type
            q2_out[:, :, agent_idx, :] = q2_coord + q2_type

        return q1_out, q2_out

    def _build_inputs(self, batch, t=None):
        bs = batch.batch_size
        max_t = batch.max_seq_length if t is None else 1
        ts = slice(None) if t is None else slice(t, t + 1)

        pieces = [
            batch["state"][:, ts].unsqueeze(2).expand(-1, -1, self.n_agents, -1)
        ]
        if getattr(self.args, "obs_individual_obs", True):
            pieces.append(batch["obs"][:, ts])
        if getattr(self.args, "obs_agent_id", True):
            pieces.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(bs, max_t, -1, -1)
            )

        return th.cat(pieces, dim=-1), bs, max_t

    def _get_input_shape(self, scheme):
        input_shape = scheme["state"]["vshape"]
        if getattr(self.args, "obs_individual_obs", True):
            input_shape += scheme["obs"]["vshape"]
        if getattr(self.args, "obs_agent_id", True):
            input_shape += self.n_agents
        return input_shape
