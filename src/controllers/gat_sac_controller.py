import torch as th
import torch.nn.functional as F
from torch.distributions import Categorical

from components.action_selectors import REGISTRY as action_REGISTRY
from modules.agents import REGISTRY as agent_REGISTRY


class GATSACMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)
        self.agent_output_type = args.agent_output_type
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env=0, bs=slice(None), test_mode=False):
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        if self._use_random_warmup(t_env, test_mode):
            return self._sample_random_actions(avail_actions[bs])
        agent_probs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        return self.action_selector.select_action(
            agent_probs[bs], avail_actions[bs], t_env, test_mode=test_mode
        )

    def forward(self, ep_batch, t, test_mode=False, detach_encoder=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_logits, self.hidden_states = self.agent(
            agent_inputs, self.hidden_states, detach_encoder=detach_encoder
        )
        masked_logits = self._mask_logits(agent_logits, avail_actions)
        agent_probs = F.softmax(masked_logits, dim=-1)
        return agent_probs

    def get_log_probs(self, ep_batch, t, detach_encoder=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_logits, self.hidden_states = self.agent(
            agent_inputs, self.hidden_states, detach_encoder=detach_encoder
        )
        masked_logits = self._mask_logits(agent_logits, avail_actions)
        probs = F.softmax(masked_logits, dim=-1)
        log_probs = th.log(probs + 1e-10)
        return log_probs, probs

    def _mask_logits(self, agent_logits, avail_actions):
        masked_logits = agent_logits.clone()
        masked_logits[avail_actions == 0] = -1e10
        return masked_logits

    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, -1, -1)

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
            th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage)
        )

    def build_inputs(self, batch, t):
        return self._build_inputs(batch, t)

    def encode(self, ep_batch, t):
        agent_inputs = self._build_inputs(ep_batch, t)
        return self.agent.encode(agent_inputs)

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
                th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1)
            )

        return th.cat(inputs, dim=-1)

    def _get_input_shape(self, scheme):
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape

    def _use_random_warmup(self, t_env, test_mode):
        return (
            not test_mode
            and getattr(self.args, "lgdd_random_warmup", False)
            and t_env < getattr(self.args, "lgdd_pretrain_steps", 0)
        )

    def _sample_random_actions(self, avail_actions):
        probs = avail_actions.float()
        probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-8)
        actions = Categorical(probs).sample().long()
        if getattr(self.action_selector, "save_probs", False):
            return actions, probs
        return actions
