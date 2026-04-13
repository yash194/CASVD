import math

import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LayerNorm

from utils.th_utils import orthogonal_init_


class LocalEntityGAT(nn.Module):
    def __init__(self, hidden_dim, n_heads=4):
        super(LocalEntityGAT, self).__init__()
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"

        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, query_node, entity_nodes, entity_mask):
        bs, n_agents, n_entities, hidden_dim = entity_nodes.shape

        query = self.q_proj(query_node).view(bs, n_agents, self.n_heads, self.head_dim)
        keys = self.k_proj(entity_nodes).view(
            bs, n_agents, n_entities, self.n_heads, self.head_dim
        )
        values = self.v_proj(entity_nodes).view(
            bs, n_agents, n_entities, self.n_heads, self.head_dim
        )

        query = query.unsqueeze(2)
        logits = (query * keys).sum(dim=-1) / math.sqrt(self.head_dim)
        logits = logits.permute(0, 1, 3, 2)
        logits = logits.masked_fill(~entity_mask.unsqueeze(2), -1e9)

        attn = th.softmax(logits, dim=-1)
        values = values.permute(0, 1, 3, 2, 4)
        aggregated = (attn.unsqueeze(-1) * values).sum(dim=-2)
        aggregated = aggregated.reshape(bs, n_agents, hidden_dim)
        return self.out_proj(aggregated) + query_node


class TeamGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, n_heads=4):
        super(TeamGATLayer, self).__init__()
        self.n_heads = n_heads
        self.head_dim = out_dim // n_heads
        assert out_dim % n_heads == 0, "out_dim must be divisible by n_heads"

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        self.a_src = nn.Parameter(th.zeros(n_heads, self.head_dim))
        self.a_dst = nn.Parameter(th.zeros(n_heads, self.head_dim))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))
        self.leaky_relu = nn.LeakyReLU(0.2)

        # Self-residual is added when input/output dims match.  Without it
        # the agent-to-agent aggregation can dilute or completely erase each
        # agent's own state — which is the most important signal for action
        # selection in SMACv2 (own health, cooldown, shoot range).  The
        # additive-attention formulation does not guarantee a high self-weight
        # in the softmax, so a learned-zero correction from the team would
        # otherwise wipe the own-state signal entering the GRU.
        self.use_residual = (in_dim == out_dim)

    def forward(self, x, alive_mask=None):
        """Agent-to-agent GAT with optional alive-agent masking.

        Args:
            x:          [bs, n_agents, in_dim] local summaries per agent.
            alive_mask: [bs, n_agents] bool, True where the agent is alive.
                        When provided, dead agents are masked out of the KEY
                        dimension so alive queries never attend to their
                        LayerNorm-β ghost embeddings.  If None, no masking.
        """
        bs, n_agents, _ = x.shape
        h = self.W(x).view(bs, n_agents, self.n_heads, self.head_dim)
        e_src = (h * self.a_src).sum(dim=-1)
        e_dst = (h * self.a_dst).sum(dim=-1)
        attn_logits = self.leaky_relu(e_src.unsqueeze(2) + e_dst.unsqueeze(1))
        # attn_logits: [bs, n_agents_q, n_agents_k, n_heads]

        if alive_mask is not None:
            # Self-loop fallback: every query is always allowed to attend to
            # itself.  Prevents softmax degeneracy at padding timesteps where
            # every agent's obs is zero and alive_mask is all False — without
            # this, every row would be all -1e9 and softmax would produce NaN.
            # These padding rows are later killed by the TD-loss filled mask,
            # so the self-only attention output is harmless.
            self_keep = th.eye(n_agents, dtype=th.bool, device=x.device).unsqueeze(0)
            key_valid = alive_mask.unsqueeze(1) | self_keep  # [bs, n_q, n_k]
            attn_logits = attn_logits.masked_fill(
                ~key_valid.unsqueeze(-1), -1e9
            )

        attn_weights = F.softmax(attn_logits, dim=2)
        attn_w = attn_weights.permute(0, 3, 1, 2)
        h_perm = h.permute(0, 2, 1, 3)
        out = th.matmul(attn_w, h_perm)
        out = out.permute(0, 2, 1, 3).contiguous()
        out = out.view(bs, n_agents, -1)

        # Self-residual: preserves each agent's own embedding through the
        # team aggregation.  Linear residual (no post-activation) — adding
        # ELU here would clamp destructive cancellations to zero and
        # asymmetrically compress large negative team corrections, the same
        # bug we fixed in the LocalEntityGAT residual path.
        if self.use_residual:
            out = out + x
        return out


class IndividualAgentNet(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_actions,
        gru_input_dim=None,
        use_rnn=True,
        use_layer_norm=False,
    ):
        super(IndividualAgentNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.use_rnn = use_rnn
        self.use_layer_norm = use_layer_norm
        gru_input_dim = gru_input_dim or hidden_dim
        if use_rnn:
            self.rnn = nn.GRUCell(gru_input_dim, hidden_dim)
        else:
            self.rnn = nn.Linear(gru_input_dim, hidden_dim)
        if self.use_layer_norm:
            self.layer_norm = LayerNorm(hidden_dim)
        self.policy_head = nn.Linear(hidden_dim, n_actions)

    def forward(self, x, hidden_state):
        if self.use_rnn:
            next_hidden = self.rnn(x, hidden_state)
        else:
            next_hidden = F.relu(self.rnn(x), inplace=True)
        if self.use_layer_norm:
            q_values = self.policy_head(self.layer_norm(next_hidden))
        else:
            q_values = self.policy_head(next_hidden)
        return q_values, next_hidden


class GATNSAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(GATNSAgent, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.hidden_dim = getattr(args, "hidden_dim", getattr(args, "rnn_hidden_dim"))
        self.input_shape = input_shape

        required_fields = [
            "obs_move_feats_size",
            "obs_enemy_feats_size",
            "obs_ally_feats_size",
            "obs_own_feats_size",
        ]
        missing_fields = [field for field in required_fields if not hasattr(args, field)]
        if missing_fields:
            raise ValueError(
                "GATNSAgent requires observation component metadata in args. Missing: {}".format(
                    ", ".join(missing_fields)
                )
            )

        self.move_feats_dim = args.obs_move_feats_size
        self.n_enemies, self.enemy_feat_dim = args.obs_enemy_feats_size
        self.n_allies, self.ally_feat_dim = args.obs_ally_feats_size
        self.own_feat_dim = args.obs_own_feats_size
        self.extra_input_dim = input_shape - (
            self.move_feats_dim
            + self.n_enemies * self.enemy_feat_dim
            + self.n_allies * self.ally_feat_dim
            + self.own_feat_dim
        )
        if self.extra_input_dim < 0:
            raise ValueError("Input shape is smaller than reconstructed observation components")

        n_heads = getattr(args, "n_heads", 4)
        use_rnn = getattr(args, "use_rnn", True)
        use_layer_norm = getattr(args, "use_layer_norm", False)

        # Separate projectors: own_encoder captures intrinsic agent state (health,
        # shield, unit type, attack_prob); move_encoder captures positional
        # constraints (pathing-grid availability).  Summing in hidden_dim space
        # lets each subspace specialise independently rather than forcing a single
        # linear layer to entangle "I am healthy" with "I am near a wall".
        self.own_encoder = nn.Linear(self.own_feat_dim, self.hidden_dim)
        self.move_encoder = nn.Linear(self.move_feats_dim, self.hidden_dim)
        self.enemy_encoder = nn.Linear(self.enemy_feat_dim, self.hidden_dim)
        self.ally_encoder = nn.Linear(self.ally_feat_dim, self.hidden_dim)
        self.local_entity_gat = LocalEntityGAT(self.hidden_dim, n_heads=n_heads)
        self.extra_encoder = None
        if self.extra_input_dim > 0:
            self.extra_encoder = nn.Linear(self.extra_input_dim, self.hidden_dim)
        self.team_gat = TeamGATLayer(self.hidden_dim, self.hidden_dim, n_heads=n_heads)
        # GRU receives [local_summary, team_summary] concatenated (2 * hidden_dim)
        # so Q-head sees both local tactical detail and team coordination context.
        self.shared_agent = IndividualAgentNet(
            self.hidden_dim,
            self.n_actions,
            gru_input_dim=self.hidden_dim * 2,
            use_rnn=use_rnn,
            use_layer_norm=use_layer_norm,
        )
        if use_layer_norm:
            self.entity_norm = LayerNorm(self.hidden_dim)
            self.team_norm = LayerNorm(self.hidden_dim)

        if getattr(args, "use_orthogonal", False):
            # encoder_gain controls entity/team encoder layers.
            # A meaningful gain (1.0) is essential so LGDD receives
            # non-trivial latents to predict — gain=0.01 makes latents
            # near-zero, which LGDD trivially predicts, causing a false
            # "coordinated" signal that collapses alpha prematurely.
            encoder_gain = getattr(args, "encoder_gain", getattr(args, "gain", 1.0))
            # q_head_gain controls the policy/Q output head.
            # A small gain (0.01) keeps initial Q-values near-uniform,
            # giving unbiased exploration before Q-values are meaningful.
            q_head_gain = getattr(args, "q_head_gain", getattr(args, "gain", 1.0))
            for module in [
                self.own_encoder,
                self.move_encoder,
                self.enemy_encoder,
                self.ally_encoder,
                self.local_entity_gat.out_proj,
            ]:
                orthogonal_init_(module, gain=encoder_gain)
            if self.extra_encoder is not None:
                orthogonal_init_(self.extra_encoder, gain=encoder_gain)
            nn.init.orthogonal_(self.team_gat.W.weight, gain=encoder_gain)
            orthogonal_init_(self.shared_agent.policy_head, gain=q_head_gain)

    def init_hidden(self):
        return th.zeros(self.n_agents, self.hidden_dim, device=self.own_encoder.weight.device)

    def encode(self, inputs):
        if inputs.dim() != 3:
            raise ValueError("Expected inputs of shape (bs, n_agents, input_dim)")

        bs = inputs.size(0)
        raw_obs = inputs[:, :, : self.input_shape - self.extra_input_dim]
        extras = inputs[:, :, self.input_shape - self.extra_input_dim :] if self.extra_input_dim > 0 else None

        move_feats, enemy_feats, ally_feats, own_feats = self._split_obs(raw_obs)

        # Dead-agent detection: SMACv2's get_obs_agent returns an all-zero
        # vector for units that are dead at the current timestep.  raw_obs
        # excludes the extras (agent_id one-hot, last_action one-hot), which
        # are always non-zero, so `raw_obs.abs().sum(-1) > 0` reliably
        # identifies alive agents on a per-(batch,agent) basis.  This mask is
        # forwarded to TeamGAT so that alive queries ignore dead agents'
        # LayerNorm-β ghost embeddings in the agent-to-agent attention pool.
        alive_mask = raw_obs.abs().sum(dim=-1) > 0   # [bs, n_agents] bool

        # Project intrinsic state and positional constraints separately, then
        # sum — each linear layer specialises in its own feature subspace.
        # ELU (not ReLU) preserves negative activations so signed features
        # like Δx, Δy ∈ [-1, +1] in ally/enemy encodings survive the first
        # nonlinearity instead of being half-gated to zero.
        self_node = F.elu(
            self.own_encoder(own_feats) + self.move_encoder(move_feats)
        )

        entity_nodes = [self_node.unsqueeze(2)]
        entity_masks = [th.ones(bs, self.n_agents, 1, dtype=th.bool, device=inputs.device)]

        if self.n_allies > 0:
            ally_nodes = F.elu(self.ally_encoder(ally_feats))
            entity_nodes.append(ally_nodes)
            entity_masks.append(ally_feats.abs().sum(dim=-1) > 0)

        if self.n_enemies > 0:
            enemy_nodes = F.elu(self.enemy_encoder(enemy_feats))
            entity_nodes.append(enemy_nodes)
            entity_masks.append(enemy_feats.abs().sum(dim=-1) > 0)

        entity_nodes = th.cat(entity_nodes, dim=2)
        entity_mask = th.cat(entity_masks, dim=2)

        # LocalEntityGAT returns `out_proj(aggregated) + query_node` — a linear
        # residual. We deliberately do NOT apply ELU on top: (1) if the attention
        # correction destructively cancels the query, ELU would clamp the sum to
        # zero and erase the local context entirely; (2) ELU asymmetrically
        # compresses large negative residuals toward its -α asymptote, capping
        # the downward correction range the attention can learn. Post-norm
        # (LayerNorm only) is the transformer-standard way to stabilise the
        # residual output without clipping its expressivity. Softmax inside the
        # attention already provides the nonlinearity for this sublayer.
        local_summary = self.local_entity_gat(self_node, entity_nodes, entity_mask)
        if getattr(self.args, "use_layer_norm", False):
            local_summary = self.entity_norm(local_summary)

        if self.extra_encoder is not None:
            # ELU for consistency with entity encoders — keeps negative learned
            # features alive in the residual add onto local_summary.
            local_summary = local_summary + F.elu(self.extra_encoder(extras))

        # TeamGAT now applies an internal self-residual when in_dim==out_dim,
        # so its output already contains `local_summary`.  We deliberately do
        # NOT wrap it in F.elu — a post-residual activation would either
        # clamp destructive cancellations to zero or asymmetrically compress
        # large negative corrections, mirroring the LocalEntityGAT fix.
        # LayerNorm (post-norm transformer style) stabilises the residual sum
        # without clipping its expressivity.
        team_summary = self.team_gat(local_summary, alive_mask=alive_mask)
        if getattr(self.args, "use_layer_norm", False):
            team_summary = self.team_norm(team_summary)

        return {
            "local_summary": local_summary,
            "team_summary": team_summary,
        }

    def forward(self, inputs, hidden_state, detach_encoder=False):
        logits, next_hidden, _ = self.forward_with_latents(inputs, hidden_state, detach_encoder)
        return logits, next_hidden

    def forward_with_latents(self, inputs, hidden_state, detach_encoder=False):
        """Forward pass that also returns encoder latents for LGDD.

        Returns:
            logits:      [B, n_agents, n_actions]
            next_hidden: [B, n_agents, hidden_dim]
            latents:     {"local_summary": [B, n_agents, hidden_dim],
                          "team_summary":  [B, n_agents, hidden_dim]}
        """
        latents = self.encode(inputs)
        local_summary = latents["local_summary"]
        team_summary = latents["team_summary"]
        if detach_encoder:
            local_summary = local_summary.detach()
            team_summary = team_summary.detach()

        bs = inputs.size(0)
        # GRU input: [local_summary, team_summary] — both tactical and coordination info
        gru_input = th.cat([local_summary, team_summary], dim=-1)
        flat_input = gru_input.reshape(bs * self.n_agents, self.hidden_dim * 2)
        flat_hidden = hidden_state.reshape(bs * self.n_agents, self.hidden_dim)

        flat_logits, flat_next_hidden = self.shared_agent(flat_input, flat_hidden)

        logits = flat_logits.view(bs, self.n_agents, -1)
        next_hidden = flat_next_hidden.view(bs, self.n_agents, -1)
        return logits, next_hidden, latents

    def _split_obs(self, raw_obs):
        offset = 0
        move_feats = raw_obs[:, :, offset : offset + self.move_feats_dim]
        offset += self.move_feats_dim

        enemy_total = self.n_enemies * self.enemy_feat_dim
        enemy_feats = raw_obs[:, :, offset : offset + enemy_total]
        enemy_feats = enemy_feats.view(raw_obs.size(0), self.n_agents, self.n_enemies, self.enemy_feat_dim)
        offset += enemy_total

        ally_total = self.n_allies * self.ally_feat_dim
        ally_feats = raw_obs[:, :, offset : offset + ally_total]
        ally_feats = ally_feats.view(raw_obs.size(0), self.n_agents, self.n_allies, self.ally_feat_dim)
        offset += ally_total

        own_feats = raw_obs[:, :, offset : offset + self.own_feat_dim]
        return move_feats, enemy_feats, ally_feats, own_feats
