"""SYNERGOS Phase 1 smoke test.

Builds the IQN agent, distributional mixer, CVaR-Soft selector with realistic
SMACv2 Protoss 5v5 dimensions, runs synthetic forward + backward, asserts all
shapes and gradients line up.  Does not need the SMACv2 env or sacred.

Run from the src/ directory:  python test_synergos_phase1_smoke.py
"""
import sys
from types import SimpleNamespace

import torch as th


def _fake_args():
    """Realistic Protoss-5v5 dimensions, agnostic to actual env launch."""
    return SimpleNamespace(
        # Team / action
        n_agents=5,
        n_actions=11,
        # GAT trunk
        hidden_dim=64,
        rnn_hidden_dim=64,
        n_heads=4,
        use_rnn=True,
        use_layer_norm=True,
        use_orthogonal=True,
        encoder_gain=1.0,
        q_head_gain=0.1,
        # SMACv2 obs partition
        obs_move_feats_size=4,
        obs_enemy_feats_size=(5, 7),
        obs_ally_feats_size=(4, 7),
        obs_own_feats_size=4,
        # Mixer
        state_shape=(80,),
        mixing_embed_dim=32,
        hypernet_embed=64,
        # Distributional / CVaR
        n_quantiles=8,
        huber_kappa=1.0,
        cvar_beta_start=1.0,
        cvar_beta_end=0.25,
        cvar_anneal_start=0,
        cvar_anneal_end=1_000_000,
        # Soft-QMIX
        entropy_coef=0.03,
    )


def main():
    args = _fake_args()
    B, T, N, A, K = 4, 10, args.n_agents, args.n_actions, args.n_quantiles
    state_dim = int(args.state_shape[0])

    move = args.obs_move_feats_size
    ne, ed = args.obs_enemy_feats_size
    na, ad = args.obs_ally_feats_size
    own = args.obs_own_feats_size
    obs_dim = move + ne * ed + na * ad + own
    extras = N + A      # agent_id one-hot + last_action one-hot
    input_shape = obs_dim + extras

    # ── Agent ──
    from modules.agents.iqn_gat_ns_agent import IQNGATNSAgent
    agent = IQNGATNSAgent(input_shape, args)
    expected_head = A * K
    actual_head = agent.shared_agent.policy_head.out_features
    assert actual_head == expected_head, (actual_head, expected_head)
    print(f"agent: K={agent.K}, head out={actual_head}, ok")

    inputs = th.randn(B, N, input_shape)
    hidden = th.zeros(B, N, args.hidden_dim)
    z, next_hidden = agent.forward(inputs, hidden)
    assert z.shape == (B, N, A, K), z.shape
    assert next_hidden.shape == (B, N, args.hidden_dim), next_hidden.shape
    print(f"agent.forward: z {tuple(z.shape)}, hidden {tuple(next_hidden.shape)}, ok")

    # ── Mixer ──
    from modules.mixers.dist_soft_mix import DistSoftMixer
    mixer = DistSoftMixer(args)
    states_T = th.randn(B, T, state_dim)
    z_full = th.randn(B, T, N, A, K, requires_grad=True)
    z_g = mixer.func_g_dist(z_full, states_T)
    assert z_g.shape == (B, T, N, A, K), z_g.shape
    z_gf = mixer.func_f_dist(z_g, states_T)
    assert z_gf.shape == (B, T, N, A, K), z_gf.shape
    z4 = th.randn(B, T, N, K, requires_grad=True)
    z4f = mixer.func_f_dist(z4, states_T)
    assert z4f.shape == (B, T, N, K), z4f.shape
    print(f"mixer.func_g_dist {tuple(z_g.shape)}, func_f_dist 5d {tuple(z_gf.shape)}, 4d {tuple(z4f.shape)}, ok")

    # gradient through both:
    loss = z_gf.sum() + z4f.sum()
    loss.backward()
    assert z_full.grad is not None and z4.grad is not None
    print("mixer gradients flow back to inputs and quantile dim")

    # ── Selector ──
    from components.action_selectors import IQNCVaRSoftPolicySelector
    selector = IQNCVaRSoftPolicySelector(args)
    avail = th.ones(B, N, A, dtype=th.long)
    z_now = th.randn(B, N, A, K)
    state_now = th.randn(B, state_dim)

    picked = selector.select_action(
        z_now, avail, t_env=500_000, test_mode=False, mixer=mixer, states=state_now,
    )
    assert picked.shape == (B, N), picked.shape
    assert (picked >= 0).all() and (picked < A).all()
    print(f"selector train-mode: picked {tuple(picked.shape)}, range ok, "
          f"β_t={selector._current_cvar_beta(500_000):.3f}")

    picked_test = selector.select_action(z_now, avail, t_env=500_000, test_mode=True)
    assert picked_test.shape == (B, N)
    print(f"selector test-mode (greedy mean-Q): picked {tuple(picked_test.shape)}, ok")

    # CVaR helper sanity
    z_test = th.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]])  # K=8
    cvar_full = IQNCVaRSoftPolicySelector.cvar_value(z_test, beta=1.0)   # mean = 4.5
    cvar_quart = IQNCVaRSoftPolicySelector.cvar_value(z_test, beta=0.25) # bottom 2 → mean(1,2)=1.5
    assert abs(cvar_full.item() - 4.5) < 1e-5, cvar_full
    assert abs(cvar_quart.item() - 1.5) < 1e-5, cvar_quart
    print(f"CVaR helper: β=1.0 → {cvar_full.item():.2f} (mean), "
          f"β=0.25 → {cvar_quart.item():.2f} (lower-quartile mean), ok")

    # ── Quantile Huber loss ──
    from learners.synergos_learner import SynergosLearner
    # Inline-test the loss function without instantiating a full learner:
    loss_fn = SynergosLearner._quantile_huber_loss.__get__(
        SimpleNamespace(taus=th.tensor([(2*k-1)/(2*K) for k in range(1, K+1)]),
                        huber_kappa=1.0)
    )
    pred = th.randn(B, T - 1, K, requires_grad=True)
    target = th.randn(B, T - 1, K)
    mask = th.ones(B, T - 1, 1)
    L = loss_fn(pred, target, mask)
    L.backward()
    assert pred.grad is not None and pred.grad.abs().sum().item() > 0
    print(f"quantile Huber loss: {L.item():.4f}, grad flows, ok")

    # CVaR β annealing
    sel = IQNCVaRSoftPolicySelector(args)
    b0  = sel._current_cvar_beta(0)
    b_mid = sel._current_cvar_beta(500_000)
    b1  = sel._current_cvar_beta(2_000_000)
    assert b0 == 1.0 and b1 == 0.25 and 0.25 < b_mid < 1.0, (b0, b_mid, b1)
    print(f"CVaR β anneal: t=0 → {b0}, t=500K → {b_mid:.3f}, t=2M → {b1}, ok")

    print("\n" + "─" * 64)
    print("✅ ALL SYNERGOS PHASE 1 SHAPE / GRADIENT CHECKS PASSED")
    print("─" * 64)


if __name__ == "__main__":
    main()
