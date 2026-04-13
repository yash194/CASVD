import pytest
import torch as th

from modules.agents.gat_ns_agent import IndividualAgentNet


def test_bounded_q_head_stays_within_scale():
    net = IndividualAgentNet(
        hidden_dim=8,
        n_actions=5,
        use_rnn=False,
        use_q_output_tanh=True,
        q_output_scale=2.0,
    )
    with th.no_grad():
        net.rnn.weight.fill_(5.0)
        net.rnn.bias.fill_(5.0)
        net.policy_head.weight.fill_(5.0)
        net.policy_head.bias.fill_(5.0)

    inputs = th.full((4, 8), 3.0)
    hidden = th.zeros(4, 8)
    q_values, _ = net(inputs, hidden)

    assert th.all(q_values <= 2.0 + 1e-6)
    assert th.all(q_values >= -2.0 - 1e-6)


def test_bounded_q_head_is_locally_linear_near_zero():
    scale = 2.0
    net = IndividualAgentNet(
        hidden_dim=1,
        n_actions=1,
        use_rnn=False,
        use_q_output_tanh=True,
        q_output_scale=scale,
    )
    with th.no_grad():
        net.rnn.weight.fill_(1.0)
        net.rnn.bias.zero_()
        net.policy_head.weight.fill_(1.0)
        net.policy_head.bias.zero_()

    small_inputs = th.tensor([[0.01], [0.02], [0.05]])
    hidden = th.zeros(3, 1)
    q_values, _ = net(small_inputs, hidden)
    rel_error = (q_values.squeeze(-1) - small_inputs.squeeze(-1)).abs()

    assert th.all(rel_error < 1e-4)


def test_unbounded_q_head_matches_raw_policy_head():
    net = IndividualAgentNet(
        hidden_dim=3,
        n_actions=2,
        use_rnn=False,
        use_q_output_tanh=False,
    )
    with th.no_grad():
        net.rnn.weight.copy_(th.eye(3))
        net.rnn.bias.zero_()
        net.policy_head.weight.copy_(th.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        net.policy_head.bias.copy_(th.tensor([0.5, -0.25]))

    inputs = th.tensor([[1.0, 2.0, 3.0]])
    hidden = th.zeros(1, 3)
    q_values, _ = net(inputs, hidden)

    expected = th.tensor([[1.5, 1.75]])
    assert th.allclose(q_values, expected)


def test_invalid_q_output_scale_raises():
    with pytest.raises(ValueError, match="q_output_scale must be > 0"):
        IndividualAgentNet(
            hidden_dim=4,
            n_actions=2,
            use_q_output_tanh=True,
            q_output_scale=0.0,
        )
