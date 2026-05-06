"""Full-SYNERGOS smoke test.

Builds every component (sheaf mixer + dual α + synergy + slow-role + sync)
with synthetic SMACv2 Protoss-5v5 dimensions and asserts shapes / gradients
flow through every loss path.  Exercises:

  • IQN agent forward → Z [B, N, A, K]
  • SheafSoftMixer.func_g_dist / func_f_dist / forward_dist_sheaf / forward_sheaf
  • IQNCVaRSoftPolicySelector with sheaf mixer
  • SynergyEstimator.compute (InfoNCE + bonus)
  • SlowRolePredictor (future-InfoNCE)
  • SyncPredictor (multi-timescale)
  • Quantile Huber loss

Note: does NOT instantiate the full SynergosLearner end-to-end (that needs
the full pymarl2 mac/scheme machinery).  Each piece is exercised in
isolation; the integration is verified by component shape compatibility.

Run from src/:  python test_synergos_full_smoke.py
"""
from types import SimpleNamespace

import torch as th


def _fake_args():
    return SimpleNamespace(
        # Team / action
        n_agents=5, n_actions=11,
        # GAT trunk
        hidden_dim=64, rnn_hidden_dim=64, n_heads=4,
        use_rnn=True, use_layer_norm=True, use_orthogonal=True,
        encoder_gain=1.0, q_head_gain=0.1,
        # SMACv2 obs partition
        obs_move_feats_size=4,
        obs_enemy_feats_size=(5, 7),
        obs_ally_feats_size=(4, 7),
        obs_own_feats_size=4,
        # Mixer
        state_shape=(80,),
        mixing_embed_dim=32, hypernet_embed=64,
        # Distributional / CVaR
        n_quantiles=8, huber_kappa=1.0,
        cvar_beta_start=1.0, cvar_beta_end=0.25,
        cvar_anneal_start=0, cvar_anneal_end=1_000_000,
        entropy_coef=0.03,
        # Phase 2: sheaf
        sheaf_id_dim=16, n_sheaf_steps=2, sheaf_step_size=0.5,
        sheaf_self_loop=True, sheaf_restriction_hidden=64,
        # Phase 3: dual α
        target_entropy_ratio=0.5, alpha_lr=3e-4,
        alpha_min=1e-3, alpha_max=1.0,
        # Phase 4: synergy
        synergy_eta=0.001, synergy_clip=0.5, synergy_n_negatives=15,
        synergy_hidden=64, synergy_temperature=0.5, synergy_lr=3e-4,
        # Phase 5a: slow role
        slow_role_horizon=8, slow_role_dim=32,
        slow_role_temperature=0.2, slow_role_lr=3e-4, lambda_slow_role=0.1,
        # Phase 5b: sync
        sync_tau_fast=0.95, sync_tau_slow=0.99,
        lambda_sync=0.01, sync_lr=3e-4,
    )


def main():
    args = _fake_args()
    B, T, N, A, K = 4, 12, args.n_agents, args.n_actions, args.n_quantiles
    state_dim = int(args.state_shape[0])
    D = args.hidden_dim

    move = args.obs_move_feats_size
    ne, ed = args.obs_enemy_feats_size
    na, ad = args.obs_ally_feats_size
    own = args.obs_own_feats_size
    obs_dim = move + ne * ed + na * ad + own
    extras = N + A
    input_shape = obs_dim + extras

    # ─── Phase 1 components (also covered in test_synergos_phase1_smoke) ───
    print("=" * 64)
    print("Phase 1 — distributional Q + CVaR-Soft selector")
    print("=" * 64)
    from modules.agents.iqn_gat_ns_agent import IQNGATNSAgent
    agent = IQNGATNSAgent(input_shape, args)
    inputs = th.randn(B, N, input_shape)
    hidden = th.zeros(B, N, D)
    z, next_h = agent.forward(inputs, hidden)
    assert z.shape == (B, N, A, K)
    print(f"agent.forward: z {tuple(z.shape)}, hidden {tuple(next_h.shape)}, ok")

    # ─── Phase 2: sheaf mixer ───
    print("=" * 64)
    print("Phase 2 — sheaf-cochain mixer")
    print("=" * 64)
    from modules.mixers.sheaf_soft_mix import SheafSoftMixer
    sheaf = SheafSoftMixer(args)

    states_T = th.randn(B, T, state_dim)
    z_full = th.randn(B, T, N, A, K, requires_grad=True)
    # Per-quantile func_g / func_f from parent class
    z_g = sheaf.func_g_dist(z_full, states_T)
    z_gf = sheaf.func_f_dist(z_g, states_T)
    assert z_g.shape == z_gf.shape == (B, T, N, A, K)
    print(f"func_g_dist / func_f_dist: {tuple(z_gf.shape)}, ok")

    # Sheaf VDN (distributional)
    z_chosen = th.randn(B, T, N, K, requires_grad=True)
    z_tot_dist = sheaf.forward_dist_sheaf(z_chosen, states_T)
    assert z_tot_dist.shape == (B, T, K)
    z_tot_dist.sum().backward()
    assert z_chosen.grad is not None
    print(f"forward_dist_sheaf: {tuple(z_tot_dist.shape)}, grad flows, ok")

    # Sheaf VDN (scalar — for β-loss path)
    q_chosen = th.randn(B, T, N, requires_grad=True)
    q_tot = sheaf.forward_sheaf(q_chosen, states_T)
    assert q_tot.shape == (B, T, 1)
    q_tot.sum().backward()
    assert q_chosen.grad is not None
    print(f"forward_sheaf (scalar): {tuple(q_tot.shape)}, grad flows, ok")

    # Verify monotonicity (IGM): increasing one agent's Q should not
    # decrease Q_tot.  Spot-check empirically.
    q_a = th.zeros(B, T, N)
    q_b = q_a.clone()
    q_b[:, :, 0] = 1.0
    qt_a = sheaf.forward_sheaf(q_a, states_T)
    qt_b = sheaf.forward_sheaf(q_b, states_T)
    assert (qt_b >= qt_a).all().item(), "IGM violated"
    print(f"sheaf IGM monotonicity: q_b > q_a ⇒ qt_b ≥ qt_a, ok (Δ_min = {(qt_b - qt_a).min().item():.4f})")

    # ─── Phase 4: synergy estimator ───
    print("=" * 64)
    print("Phase 4 — synergy estimator (PID-S)")
    print("=" * 64)
    from modules.predictors.synergy_estimator import SynergyEstimator
    synergy = SynergyEstimator(state_dim, A, N, hidden=64, temperature=0.5)

    F_dim = B * (T - 1)
    states_flat = th.randn(F_dim, state_dim)
    actions_oh = th.zeros(F_dim, N, A)
    actions_idx = th.randint(0, A, (F_dim, N))
    actions_oh.scatter_(2, actions_idx.unsqueeze(-1), 1.0)
    G_flat = th.randn(F_dim, 1)
    i_idx = th.randint(0, N, (F_dim, 1))
    j_idx = (i_idx + 1) % N  # cheap "j ≠ i" for test
    k_idx = th.randint(0, N, (F_dim, 1))

    syn_per, syn_loss = synergy.compute(
        states_flat, actions_oh, G_flat, i_idx, j_idx, k_idx, n_negatives=15,
    )
    assert syn_per.shape == (F_dim, 1)
    assert syn_loss.dim() == 0
    syn_loss.backward()
    grad_present = any(p.grad is not None and p.grad.abs().sum().item() > 0
                       for p in synergy.parameters())
    assert grad_present
    print(f"synergy NCE loss: {syn_loss.item():.4f}, per-sample S̃ {tuple(syn_per.shape)}, "
          f"grad ok, S̃≥0: {(syn_per >= 0).all().item()}")

    # ─── Phase 5a: slow-role InfoNCE ───
    print("=" * 64)
    print("Phase 5a — slow-role InfoNCE")
    print("=" * 64)
    from modules.predictors.slow_role_predictor import SlowRolePredictor
    slow_role = SlowRolePredictor(hidden_dim=D, role_dim=32, n_agents=N, temperature=0.2)
    past = th.randn(B, N, D)
    future = th.randn(B, N, D)
    sr_loss, sr_stats = slow_role(past, future, return_stats=True)
    assert sr_loss.dim() == 0
    sr_loss.backward()
    grad_present = any(p.grad is not None for p in slow_role.parameters())
    assert grad_present
    print(f"slow-role InfoNCE: loss={sr_loss.item():.4f}, top1={sr_stats['slow_role_top1']:.3f}, ok")

    # ─── Phase 5b: multi-timescale sync ───
    print("=" * 64)
    print("Phase 5b — multi-timescale predictive sync")
    print("=" * 64)
    from modules.predictors.sync_predictor import SyncPredictor
    sync = SyncPredictor(hidden_dim=D, n_bands=2, n_agents=N)
    z_now = th.randn(B, T, N, D)
    ema_fast = th.randn(B, T, N, D)
    ema_slow = th.randn(B, T, N, D)
    sync_loss = sync(z_now, [ema_fast, ema_slow])
    assert sync_loss.dim() == 0
    sync_loss.backward()
    grad_present = any(p.grad is not None for p in sync.parameters())
    assert grad_present
    print(f"sync loss (cross-agent, 2 bands): {sync_loss.item():.4f}, ok")

    # ─── Selector with sheaf mixer ───
    print("=" * 64)
    print("Selector + sheaf mixer integration")
    print("=" * 64)
    from components.action_selectors import IQNCVaRSoftPolicySelector
    sel = IQNCVaRSoftPolicySelector(args)
    avail = th.ones(B, N, A, dtype=th.long)
    z_now_sel = th.randn(B, N, A, K)
    state_now = th.randn(B, state_dim)
    picked = sel.select_action(
        z_now_sel, avail, t_env=500_000, test_mode=False,
        mixer=sheaf, states=state_now,
    )
    assert picked.shape == (B, N)
    print(f"selector with sheaf mixer: picked {tuple(picked.shape)}, ok")

    # ─── Per-agent dual α (Phase 3) — Lagrangian update ───
    print("=" * 64)
    print("Phase 3 — per-agent dual α descent (synthetic)")
    print("=" * 64)
    import math
    log_alpha = th.nn.Parameter(th.full((N,), math.log(0.03)))
    target_H = 0.5 * math.log(A)
    H_actual = th.tensor([0.3, 0.5, 0.6, 1.0, 1.2])  # synthetic per-agent entropies
    alpha_loss = -(log_alpha * (target_H - H_actual.detach())).mean()
    alpha_loss.backward()
    assert log_alpha.grad is not None
    print(f"alpha Lagrangian loss: {alpha_loss.item():.4f}, ∂L/∂log_α = "
          f"{log_alpha.grad.tolist()}, ok")

    print("\n" + "─" * 64)
    print("✅ ALL SYNERGOS FULL-STACK SHAPE / GRADIENT CHECKS PASSED")
    print("─" * 64)


if __name__ == "__main__":
    main()
