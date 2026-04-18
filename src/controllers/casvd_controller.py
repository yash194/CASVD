import torch as th
from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY


class CASVDMAC:
    """Controller for CASVD: GAT encoder with shared Q-head.

    During rollout: uses Boltzmann softmax(Q / alpha) for training,
    argmax for test — as specified in CASVD.md Math 4.
    Alpha is updated by the learner after each training step.
    For training: provides Q-values directly (no separate actor).
    Also exposes encode() for LGDD auxiliary loss.
    """

    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None
        self.mixer = None

    def set_mixer(self, mixer):
        """Called by the learner to provide the mixer for soft-policy rollout."""
        self.mixer = mixer

    def select_actions(self, ep_batch, t_ep, t_env=0, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        # SoftPolicyActionSelector needs mixer + states for func_g/func_f.
        # Standard selectors (epsilon_greedy, etc.) don't — pass only if
        # the selector accepts them to stay backward-compatible.
        if self.mixer is not None and hasattr(self.action_selector, "entropy_coef"):
            states = ep_batch["state"][:, t_ep]
            return self.action_selector.select_action(
                agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode,
                mixer=self.mixer, states=states[bs],
            )
        return self.action_selector.select_action(
            agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode,
        )

    def forward(self, ep_batch, t, test_mode=False):
        """Returns Q-values for all actions. Shape: [B, n_agents, n_actions]

        NOTE: Q-values are returned RAW (no masking). Masking for
        unavailable actions is handled by the action selector and
        the learner, not here, so that the learner sees clean Q-values
        for logsumexp / softmax computations.
        """
        agent_inputs = self._build_inputs(ep_batch, t)
        q_values, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        return q_values

    def forward_with_latents(self, ep_batch, t):
        """Returns Q-values (raw) AND encoder latents for LGDD.

        Returns:
            q_values: [B, n_agents, n_actions] (raw, unmasked)
            latents:  {"local_summary": ..., "team_summary": ...}
        """
        agent_inputs = self._build_inputs(ep_batch, t)
        q_values, self.hidden_states, latents = self.agent.forward_with_latents(
            agent_inputs, self.hidden_states
        )
        return q_values, latents

    def set_alpha(self, alpha):
        """Called by the learner to update the Boltzmann temperature.

        This connects the coordination-aware alpha to the action selection
        policy, as specified in CASVD.md.
        """
        if hasattr(self.action_selector, "alpha"):
            self.action_selector.alpha = alpha

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(
            batch_size, -1, -1
        )

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.to(getattr(self.args, "device", "cuda"))

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(
            th.load(
                "{}/agent.th".format(path),
                map_location=lambda storage, loc: storage,
            )
        )

    def build_inputs(self, batch, t):
        """Public accessor for learner to build inputs (used by LGDD)."""
        return self._build_inputs(batch, t)

    def _build_inputs(self, batch, t):
        bs = batch.batch_size
        inputs = [batch["obs"][:, t]]
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t - 1])
        if self.args.obs_agent_id:
            inputs.append(
                th.eye(self.n_agents, device=batch.device)
                .unsqueeze(0)
                .expand(bs, -1, -1)
            )
        return th.cat(inputs, dim=-1)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
