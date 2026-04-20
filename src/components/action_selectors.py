from matplotlib.pyplot import xcorr
import torch as th
from torch.distributions import Categorical
from torch.distributions.one_hot_categorical import OneHotCategorical
from .epsilon_schedules import DecayThenFlatSchedule

class GumbelSoftmax(OneHotCategorical):

    def __init__(self, logits, probs=None, temperature=1):
        super(GumbelSoftmax, self).__init__(logits=logits, probs=probs)
        self.eps = 1e-20
        self.temperature = temperature

    def sample_gumbel(self):
        U = self.logits.clone()
        U.uniform_(0, 1)
        return -th.log( -th.log( U + self.eps))

    def gumbel_softmax_sample(self):
        y = self.logits + self.sample_gumbel()
        return th.softmax( y / self.temperature, dim=-1)

    def hard_gumbel_softmax_sample(self):
        y = self.gumbel_softmax_sample()
        return (th.max(y, dim=-1, keepdim=True)[0] == y).float()

    def rsample(self):
        return self.gumbel_softmax_sample()

    def sample(self):
        return self.rsample().detach()

    def hard_sample(self):
        return self.hard_gumbel_softmax_sample()

def multinomial_entropy(logits):
    assert logits.size(-1) > 1
    return GumbelSoftmax(logits=logits).entropy()

REGISTRY = {}

class GumbelSoftmaxMultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True)
        self.save_probs = getattr(self.args, 'save_probs', False)

    def select_action(self, agent_logits, avail_actions, t_env, test_mode=False):
        masked_policies = agent_logits.clone()
        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            picked_actions = GumbelSoftmax(logits=masked_policies).sample()
            picked_actions = th.argmax(picked_actions, dim=-1).long()

        if self.save_probs:
            return picked_actions, masked_policies
        else:
            return picked_actions


REGISTRY["gumbel"] = GumbelSoftmaxMultinomialActionSelector


class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

        self.test_greedy = getattr(args, "test_greedy", True)
        self.save_probs = getattr(self.args, 'save_probs', False)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0] = 0
        masked_policies = masked_policies / (masked_policies.sum(-1, keepdim=True) + 1e-8)

        if test_mode and self.test_greedy:
            picked_actions = masked_policies.max(dim=2)[1]
        else:
            self.epsilon = self.schedule.eval(t_env)

            epsilon_action_num = (avail_actions.sum(-1, keepdim=True) + 1e-8)
            masked_policies = ((1 - self.epsilon) * masked_policies
                        + avail_actions * self.epsilon/epsilon_action_num)
            masked_policies[avail_actions == 0] = 0
            
            picked_actions = Categorical(masked_policies).sample().long()

        if self.save_probs:
            return picked_actions, masked_policies
        else:
            return picked_actions

REGISTRY["multinomial"] = MultinomialActionSelector

def categorical_entropy(probs):
    assert probs.size(-1) > 1
    return Categorical(probs=probs).entropy()


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # Assuming agent_inputs is a batch of Q-Values for each agent bav
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # Greedy action selection only
            self.epsilon  = getattr(self.args, "test_noise", 0.0)

        # mask actions that are excluded from selection
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0] = -float("inf")  # should never be selected!
        
        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector


class BoltzmannActionSelector():
    """Boltzmann (softmax) action selection for CASVD.

    During training: sample from softmax(Q / alpha) with action masking.
    During test: greedy argmax over Q-values.

    Alpha is set externally by the learner via `self.alpha`.
    """

    def __init__(self, args):
        self.args = args
        self.alpha = getattr(args, "fixed_alpha", 0.1)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # agent_inputs: [B, n_agents, n_actions] (Q-values)
        masked_q = agent_inputs.clone()
        masked_q[avail_actions == 0] = -float("inf")

        if test_mode:
            # Greedy (argmax)
            picked_actions = masked_q.max(dim=2)[1]
        else:
            # Boltzmann: sample from softmax(Q / alpha)
            alpha = max(self.alpha, 1e-4)
            logits = masked_q / alpha
            # Replace -inf/alpha = -inf with a large negative for softmax
            logits[avail_actions == 0] = -1e10
            probs = th.softmax(logits, dim=-1)
            picked_actions = Categorical(probs).sample().long()

        return picked_actions


REGISTRY["boltzmann"] = BoltzmannActionSelector


class SoftPolicyActionSelector():
    """Soft-QMIX action selection: softmax(func_f(func_g(Q)) / α) sampling.

    During training: applies mixer's func_g and func_f to raw Q-values,
    then samples from the Boltzmann soft policy.
    During test: greedy argmax on raw Q-values (no transformations).

    α supports both scalar and per-agent tensor form.  The learner
    pushes the current α vector via the controller's `set_alpha` hook
    (sets `self.alpha_vec`); when a per-agent α tensor is present it
    takes precedence over the scalar fallback.
    """

    def __init__(self, args):
        self.args = args
        self.entropy_coef = getattr(args, "entropy_coef", 0.03)
        # Per-agent α vector, shape [n_agents].  Set externally by the
        # learner via CASVDMAC.set_alpha — stays None until then, at
        # which point the scalar `entropy_coef` fallback is used.
        self.alpha_vec = None

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False,
                       mixer=None, states=None):
        if test_mode:
            masked_q = agent_inputs.clone()
            masked_q[avail_actions == 0] = -float("inf")
            picked_actions = masked_q.max(dim=2)[1]
            return picked_actions

        q = agent_inputs
        if mixer is not None and states is not None:
            q = mixer.func_g(q, states, t_env).detach()
            q = mixer.func_f(q, states, t_env).detach()

        # Per-agent α if available, else scalar.  agent_inputs is
        # [B, n_agents, n_actions]; reshape α [N] → [1, N, 1] to divide.
        if self.alpha_vec is not None:
            alpha = self.alpha_vec
            if hasattr(alpha, "to"):
                alpha = alpha.to(q.device)
                divisor = alpha.view(1, -1, 1)
            else:
                divisor = float(alpha)
            logits = q / divisor
        else:
            logits = q / self.entropy_coef
        logits[avail_actions == 0] = -float("inf")
        probs = th.softmax(logits, dim=-1)

        cdf = th.cumsum(probs, dim=-1)
        rand_idx = th.rand(probs[:, :, :1].shape, device=probs.device)
        rand_idx = th.clamp(rand_idx, 1e-6, 1 - 1e-6)
        picked_actions = th.searchsorted(cdf, rand_idx)
        return picked_actions.squeeze(-1)


REGISTRY["soft_policy"] = SoftPolicyActionSelector


class GaussianActionSelector():

    def __init__(self, args):
        self.args = args
        self.test_greedy = getattr(args, "test_greedy", True)

    def select_action(self, mu, sigma, test_mode=False):
        # Expects the following input dimensions:
        # mu: [b x a x u]
        # sigma: [b x a x u x u]
        assert mu.dim() == 3, "incorrect input dim: mu"
        assert sigma.dim() == 3, "incorrect input dim: sigma"
        sigma = sigma.view(-1, self.args.n_agents, self.args.n_actions, self.args.n_actions)

        if test_mode and self.test_greedy:
            picked_actions = mu
        else:
            dst = th.distributions.MultivariateNormal(mu.view(-1,
                                                              mu.shape[-1]),
                                                      sigma.view(-1,
                                                                 mu.shape[-1],
                                                                 mu.shape[-1]))
            try:
                picked_actions = dst.sample().view(*mu.shape)
            except Exception as e:
                a = 5
                pass
        return picked_actions


REGISTRY["gaussian"] = GaussianActionSelector