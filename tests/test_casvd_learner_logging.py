from types import SimpleNamespace

import torch as th

from learners.casvd_learner import CASVDLearner


class DummyLogger:
    def __init__(self):
        self.stats = {}

    def log_stat(self, name, value, t_env):
        self.stats[name] = (value, t_env)


class DummyMAC(th.nn.Module):
    def __init__(self, n_agents, n_actions, hidden_dim):
        super().__init__()
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim
        self.bias = th.nn.Parameter(th.tensor(0.0))

    def init_hidden(self, batch_size):
        self.hidden = th.zeros(batch_size, self.n_agents, self.hidden_dim)

    def forward(self, batch, t):
        base = th.tensor([0.2, 0.5, 0.8], device=batch.device)
        q = base.view(1, 1, self.n_actions).expand(batch.batch_size, self.n_agents, self.n_actions).clone()
        q = q + self.bias
        return q

    def forward_with_latents(self, batch, t):
        q = self.forward(batch, t)
        latents = {
            "local_summary": th.zeros(batch.batch_size, self.n_agents, self.hidden_dim, device=batch.device),
            "team_summary": th.zeros(batch.batch_size, self.n_agents, self.hidden_dim, device=batch.device),
        }
        return q, latents

    def load_state(self, other_mac):
        self.load_state_dict(other_mac.state_dict())


class FakeBatch:
    def __init__(self, data):
        self.data = data
        self.batch_size = data["reward"].shape[0]
        self.max_seq_length = data["avail_actions"].shape[1]
        self.device = data["reward"].device

    def __getitem__(self, key):
        return self.data[key]


def test_casvd_logs_agent_q_stats_and_alpha_floor_hit_rate():
    args = SimpleNamespace(
        n_agents=2,
        n_actions=3,
        state_shape=(4,),
        mixing_embed_dim=4,
        hypernet_embed=4,
        lr=0.001,
        optimizer_epsilon=1e-7,
        use_soft_values=True,
        alpha_factor_init=0.5,
        alpha_factor_min=0.0,
        alpha_factor_max=1.0,
        alpha_floor=0.3,
        use_adaptive_alpha=False,
        coord_signal_ema_tau=0.99,
        lgdd_enabled=False,
        infonce_n_negatives=15,
        infonce_temperature=0.1,
        cl_enabled=False,
        cl_distill_weight=0.0,
        cl_teacher_ema_tau=0.002,
        grad_norm_clip=10,
        gamma=0.99,
        td_lambda=0.6,
        target_update_interval_or_tau=200,
        learner_log_interval=1,
        use_orthogonal=False,
        qmix_pos_func="abs",
        device="cpu",
    )
    logger = DummyLogger()
    mac = DummyMAC(args.n_agents, args.n_actions, hidden_dim=4)
    learner = CASVDLearner(mac, scheme=None, logger=logger, args=args)

    batch = FakeBatch(
        {
            "reward": th.zeros(1, 2, 1),
            "actions": th.tensor([[[[0], [1]], [[0], [1]]]], dtype=th.long),
            "terminated": th.zeros(1, 2, 1),
            "filled": th.ones(1, 2, 1),
            "avail_actions": th.ones(1, 2, 2, 3),
            "state": th.zeros(1, 2, 4),
        }
    )

    learner.train(batch, t_env=1, episode_num=1)

    for key in [
        "agent_q_taken_mean",
        "agent_q_mean",
        "agent_q_std",
        "agent_q_max_abs",
        "alpha_floor_hit_rate",
        "q_taken_mean",
        "target_mean",
    ]:
        assert key in logger.stats

    assert logger.stats["agent_q_max_abs"][0] <= 0.8 + 1e-6
    assert 0.0 <= logger.stats["alpha_floor_hit_rate"][0] <= 1.0
