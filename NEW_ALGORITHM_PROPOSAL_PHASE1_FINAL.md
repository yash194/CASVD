# Risk-Aware Distributional Soft-QMIX for SMACv2

## Final Phase-1 Research Contribution

This document defines the final scope of the paper. The work builds on the existing `pymarl2-with-smacv2-extra` implementation, whose baseline is **CASVD: GAT-based agent encoding with Soft-QMIX value decomposition**. The paper contribution is limited to **Phase 1** of the original roadmap:

> Replace the scalar per-agent value head with an IQN-style distributional value head and use CVaR-based risk-sensitive action selection, while retaining the existing GAT encoder, Soft-QMIX mixer, entropy formulation, and CTDE training structure.

Phase 0 is retained only to establish the multi-seed baseline needed for comparison. All previously proposed later-stage additions—sheaf mixing, PID-synergy rewards, role discovery, predictive synchrony, and per-agent dual-entropy tuning—are outside the scope of this paper.

---

## 0. Executive Summary

### Problem

Soft-QMIX learns scalar expected action-values. This is potentially limiting on SMACv2 because the environment introduces substantial episode-level uncertainty through:

- randomized unit composition,
- randomized spawn configuration,
- partial observability,
- stochastic interaction outcomes,
- and variation in whether a locally reasonable action succeeds under a particular episode configuration.

A scalar value head compresses the complete return distribution into a single expectation. Two actions can therefore receive similar expected values even when one has stable outcomes and the other has a much worse lower tail. This may produce policies that perform well on average but remain brittle under difficult spawn and composition combinations.

### Proposed Contribution

We propose **Risk-Aware Distributional Soft-QMIX**, a minimal extension of the existing GAT + Soft-QMIX baseline. The method:

1. preserves the existing GAT-based decentralized agent representation;
2. replaces each scalar agent utility output with an implicit quantile distribution;
3. applies the existing monotonic Soft-QMIX mixer quantile-wise;
4. trains the return distribution using quantile Huber regression;
5. uses lower-tail CVaR values during training-time action selection;
6. retains centralized training and decentralized execution;
7. uses mean-value greedy evaluation at test time as the main reporting protocol, with CVaR-greedy evaluation included as an additional risk-sensitive analysis.

### Research Question

> Does explicitly modelling the return distribution and using lower-tail risk during action selection improve robustness, sample efficiency, and worst-case performance over scalar GAT + Soft-QMIX on procedurally randomized SMACv2 scenarios?

### Main Hypothesis

The distributional model should improve performance most clearly in the lower tail of the episode-return distribution. The expected result is not merely a higher mean win rate, but reduced sensitivity to unfavorable unit compositions and spawn configurations.

### Final Paper Scope

The paper contains only:

- a multi-seed scalar GAT + Soft-QMIX baseline;
- the IQN-based distributional extension;
- CVaR-based risk-sensitive action selection;
- ablations isolating distributional modelling from risk sensitivity;
- robustness analysis across return quantiles, spawn conditions, and unit compositions.

No claims are made for components beyond Phase 1.

---

## 1. Problem Setting and Notation

We consider a cooperative Decentralized Partially Observable Markov Decision Process:

```text
⟨N, S, {O_i}, {A_i}, P, R, γ⟩
```

where:

- `N` is the number of agents;
- `S` is the global state space available only during centralized training;
- `O_i` and `A_i` are the observation and action spaces of agent `i`;
- `P` is the environment transition function;
- `R` is the shared team reward;
- `γ ∈ [0,1)` is the discount factor.

For agent `i`, its action-observation history at time `t` is:

```text
τ_i^t = (o_i^0, a_i^0, r^0, ..., o_i^t)
```

The joint action is:

```text
a = (a_1, ..., a_N)
```

The discounted team return from time `t` is:

```text
G_t = Σ_{k=0}^∞ γ^k r_{t+k}
```

Under centralized training with decentralized execution, the learned policy factorizes as:

```text
π(a | τ) = ∏_i π_i(a_i | τ_i)
```

The global state and joint action may be used by the mixer during training, but each agent must select its own action from local information at execution time.

### 1.1 SMACv2 Characteristics Relevant to This Work

SMACv2 differs from fixed-scenario cooperative benchmarks because episode generation introduces structural variability. Depending on the selected scenario, agents face randomized:

- allied and enemy unit types;
- unit positions and spawn sectors;
- engagement geometries;
- local visibility relationships;
- action consequences under different team compositions.

This creates aleatoric uncertainty: even similar high-level decisions can produce different returns because of episode conditions that cannot be fully eliminated by additional training data.

---

## 2. Existing Baseline: GAT + Soft-QMIX

The starting implementation is CASVD, consisting of:

1. a GAT-based agent encoder;
2. per-agent recurrent or local utility estimation;
3. a Soft-QMIX monotonic value mixer;
4. entropy-regularized action selection;
5. off-policy replay and target-network learning.

### 2.1 Scalar Agent Utilities

In the baseline, each agent outputs a scalar utility for every available action:

```text
Q_i(τ_i, a_i) ∈ ℝ
```

The joint value is produced by a monotonic mixer:

```text
Q_tot(s, a) = M_s(Q_1(τ_1,a_1), ..., Q_N(τ_N,a_N))
```

with the QMIX monotonicity constraint:

```text
∂Q_tot / ∂Q_i ≥ 0
```

This supports the Individual-Global-Max property: maximizing each local utility independently is consistent with maximizing the centralized joint value represented by the mixer.

### 2.2 Limitation of the Scalar Head

The scalar utility represents only an expected return:

```text
Q_i(τ_i,a_i) = E[G_t | τ_i,a_i]
```

It does not preserve information about:

- return variance;
- multimodal outcomes;
- probability of catastrophic failure;
- lower-tail performance;
- uncertainty induced by random episode generation.

For example, two actions may both have expected return `0.6`, while one is consistently moderate and the other alternates between very high reward and severe failure. The scalar critic treats these actions as equivalent even though their robustness differs.

---

## 3. Final Contribution: Risk-Aware Distributional Soft-QMIX

The proposed method models the conditional return distribution rather than only its expectation.

For each agent and action, the scalar utility is replaced by a random return variable:

```text
Z_i(τ_i,a_i)
```

such that:

```text
Q_i(τ_i,a_i) = E[Z_i(τ_i,a_i)]
```

The distribution is represented through sampled quantiles using an Implicit Quantile Network.

### 3.1 Design Principles

The Phase-1 method intentionally changes as little of the baseline as possible.

**Kept unchanged:**

- GAT encoder;
- decentralized agent histories;
- Soft-QMIX mixer structure;
- monotonic mixing constraints;
- replay buffer;
- target-network update mechanism;
- entropy regularization;
- CTDE execution structure;
- environment and reward definition.

**Changed:**

- scalar action-value head → IQN quantile head;
- scalar TD objective → quantile Huber TD objective;
- expected-value training action score → CVaR-based risk score;
- logging → distributional and tail-risk diagnostics.

This restricted design ensures that any observed difference can be attributed primarily to distributional value learning and risk-sensitive action selection.

---

## 4. Architecture

```text
           Local observation/history τ_i
                       │
                       ▼
        ┌──────────────────────────────┐
        │ Existing GAT Agent Encoder   │
        │ h_i = Encoder_GAT(τ_i)       │
        └──────────────────────────────┘
                       │
           sampled quantile fractions τ_k
                       │
                       ▼
        ┌──────────────────────────────┐
        │ IQN Distributional Head      │
        │ Z_i(τ_i,a_i;τ_k), k=1...K    │
        └──────────────────────────────┘
                       │
          per-agent quantile utilities
                       │
                       ▼
        ┌──────────────────────────────┐
        │ Existing Soft-QMIX Mixer     │
        │ applied independently at    │
        │ each sampled quantile        │
        └──────────────────────────────┘
                       │
                       ▼
        Joint quantile samples Z_tot(τ_k)
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
      Quantile TD loss     CVaR risk score
                             for action selection
```

### 4.1 GAT Encoder

The existing GAT encoder produces an agent representation:

```text
h_i^t = φ_θ(τ_i^t)
```

No new graph-learning contribution is introduced in this paper. The GAT is treated as part of the established baseline architecture.

### 4.2 IQN Quantile Embedding

For each sampled quantile fraction `τ_k ~ Uniform(0,1)`, an IQN cosine embedding is formed:

```text
ψ(τ_k) = ReLU(W_ψ [cos(π·1·τ_k), ..., cos(π·n·τ_k)] + b_ψ)
```

The agent representation and quantile embedding are combined element-wise or through a learned fusion layer:

```text
u_i^k = h_i ⊙ ψ(τ_k)
```

The distributional head outputs one value per action:

```text
Z_i(τ_i,a_i;τ_k) = f_θ(u_i^k, a_i)
```

For `K` sampled fractions, each action therefore has `K` quantile-value samples rather than one scalar value.

### 4.3 Quantile-Wise Soft-QMIX Mixing

For every sampled quantile `τ_k`, the existing Soft-QMIX mixer receives one utility from each agent:

```text
Z_tot(s,a;τ_k)
  = M_s(
      Z_1(τ_1,a_1;τ_k),
      ...,
      Z_N(τ_N,a_N;τ_k)
    )
```

The same mixer parameters are shared across quantile samples. This preserves the baseline mixer and introduces no new heterogeneous or topological mixing mechanism.

The mean joint value is estimated by Monte Carlo integration:

```text
Q_tot(s,a) ≈ (1/K) Σ_k Z_tot(s,a;τ_k)
```

### 4.4 Risk Score Using CVaR

For lower-tail level `β`, the Conditional Value-at-Risk score is approximated from quantiles satisfying `τ_k ≤ β`:

```text
CVaR_β[Z_i(τ_i,a_i)]
  ≈ (1/|K_β|) Σ_{k:τ_k≤β} Z_i(τ_i,a_i;τ_k)
```

where `β = 1.0` recovers the approximate mean and smaller values focus increasingly on poor outcomes.

The training-time action score becomes:

```text
R_i(a_i) = CVaR_β[Z_i(τ_i,a_i)]
```

The entropy-regularized decentralized policy is then:

```text
π_i(a_i | τ_i)
  ∝ exp(R_i(a_i) / α)
```

Unavailable actions remain masked exactly as in the baseline.

### 4.5 CVaR Schedule

A practical schedule begins near risk-neutral learning and gradually increases lower-tail sensitivity:

```text
β: 1.00 → 0.75 → 0.50 → 0.25
```

The schedule prevents strong early pessimism from suppressing exploration before the return distribution is sufficiently learned.

The final schedule is a hyperparameter and must be compared against fixed-risk alternatives.

---

## 5. Learning Objective

### 5.1 Target Action Selection

For the next state, each agent selects an action using the online network's CVaR-based policy:

```text
a_i' ~ π_i^online(· | τ_i')
```

or, for the greedy Double-Q variant:

```text
a_i' = argmax_a CVaR_β[Z_i^online(τ_i',a)]
```

The target network evaluates the resulting joint action.

### 5.2 Distributional Bellman Target

For target quantile fractions `τ'_{k'}`, the target distribution samples are:

```text
y_{k'} = r
         + γ(1-done) · Z_tot^target(s',a';τ'_{k'})
         + entropy_bonus
```

The entropy term from the existing Soft-QMIX implementation is retained. Because it is a scalar regularization term, it is broadcast consistently across the target quantile samples.

CVaR is used to choose or sample the next action. The Bellman target itself retains the complete target quantile distribution rather than collapsing it to a single CVaR value.

### 5.3 Pairwise Quantile TD Errors

For current samples `τ_k` and target samples `τ'_{k'}`:

```text
δ_{k,k'} = y_{k'} - Z_tot^online(s,a;τ_k)
```

### 5.4 Quantile Huber Loss

The Huber loss is:

```text
L_κ(δ) =
  0.5δ²,                    if |δ| ≤ κ
  κ(|δ| - 0.5κ),            otherwise
```

The asymmetric quantile regression loss is:

```text
ρ_τ^κ(δ)
  = |τ - 1{δ<0}| · L_κ(δ) / κ
```

The complete distributional TD loss is:

```text
L_quantile
  = (1 / KK') Σ_k Σ_k' ρ_{τ_k}^κ(δ_{k,k'})
```

The final learner objective is:

```text
L = L_quantile + λ_β L_β
```

where `L_β` is the existing Soft-QMIX auxiliary beta loss retained for backward compatibility. No synergy, role, synchrony, sheaf, or adaptive-entropy auxiliary loss is included.

---

## 6. Training and Execution

### 6.1 Training Rollout

At each environment step:

1. encode each agent's local history using the existing GAT encoder;
2. sample `K_policy` quantile fractions;
3. estimate each action's CVaR score;
4. mask unavailable actions;
5. sample from the entropy-regularized CVaR policy;
6. store the transition in the existing replay buffer.

### 6.2 Learner Update

For every sampled replay batch:

1. sample online and target quantile fractions;
2. compute online per-agent quantile utilities;
3. mix them quantile-wise through Soft-QMIX;
4. select next actions using online CVaR values;
5. evaluate target quantiles using the target network;
6. construct pairwise quantile TD errors;
7. optimize the quantile Huber objective;
8. update the target network according to the existing schedule.

### 6.3 Test-Time Policies

Two evaluation modes should be reported.

**Primary: mean-greedy evaluation**

```text
a_i = argmax_a E_τ[Z_i(τ_i,a;τ)]
```

This measures whether distributional training improves the standard risk-neutral deployment objective.

**Secondary: CVaR-greedy evaluation**

```text
a_i = argmax_a CVaR_β[Z_i(τ_i,a)]
```

This measures performance when robust lower-tail behavior is explicitly preferred at deployment.

The primary comparison against scalar Soft-QMIX should use mean-greedy testing unless the baseline is also evaluated under an equivalent risk-sensitive decision rule.

---

## 7. Phase Structure Retained in the Paper

### Phase 0 — Multi-Seed Baseline

Run the current scalar CASVD implementation using identical environment, compute, logging, and evaluation settings.

Required baseline runs:

- scalar GAT + Soft-QMIX;
- at least 5 seeds where compute permits;
- identical training horizon for all compared methods;
- separate reporting for Protoss, Terran, and Zerg scenarios used in the study;
- evaluation at fixed environment-step intervals;
- mean, median, standard deviation, and confidence intervals.

Phase 0 is not presented as a research contribution. It is the control condition required to establish whether Phase 1 provides a real improvement.

### Phase 1 — Final Research Contribution

Add only:

- IQN distributional per-agent action-value head;
- quantile-wise Soft-QMIX mixing;
- quantile Huber Bellman loss;
- CVaR-based training action selection;
- distributional and tail-risk diagnostics.

The final paper ends at Phase 1. There is no subsequent algorithmic phase.

---

## 8. Implementation Plan for `pymarl2-with-smacv2-extra`

### 8.1 New Files

| File | Purpose |
|---|---|
| `src/modules/heads/iqn_head.py` | Quantile embedding and per-action distributional output |
| `src/learners/distributional_soft_qmix_learner.py` | Quantile target construction and quantile Huber optimization |
| `src/components/cvar_action_selector.py` | CVaR-based entropy-regularized action scoring and masking |
| `src/config/algs/distributional_soft_qmix.yaml` | Algorithm and ablation configuration |

### 8.2 Existing Files to Modify

| File | Required Change |
|---|---|
| `src/modules/agents/gat_ns_agent.py` | Return latent features suitable for the IQN head instead of immediately collapsing to scalar utilities |
| existing Soft-QMIX mixer | Accept an extra quantile dimension and apply the same monotonic mixer independently for each quantile sample |
| existing controller/MAC | Request quantile samples and pass CVaR scores to the action selector |
| existing learner registration | Register the distributional learner and configuration |
| logging utilities | Add quantile spread, CVaR, lower-tail return, and quantile-crossing diagnostics |

### 8.3 Suggested Tensor Shapes

For batch size `B`, sequence length `T`, agents `N`, actions `A`, and sampled quantiles `K`:

```text
agent_quantiles: [B, T, N, A, K]
chosen_quantiles: [B, T, N, K]
joint_quantiles:  [B, T, K]
target_quantiles: [B, T, K_target]
pairwise_delta:   [B, T, K, K_target]
```

Explicit shape assertions should be included because quantile and agent dimensions are easy to accidentally interchange.

### 8.4 Initial Hyperparameters

| Hyperparameter | Initial Value / Range |
|---|---|
| online quantiles `K` | 8 or 16 |
| target quantiles `K_target` | 8 or 16 |
| policy quantiles `K_policy` | 16 or 32 |
| cosine embedding dimension | 64 |
| Huber threshold `κ` | 1.0 |
| final CVaR level `β` | 0.25 |
| CVaR warm-up | begin at `β=1.0` |
| beta-loss coefficient `λ_β` | retain baseline value, initially 0.1 if that is the current configuration |
| optimizer | same as scalar baseline |
| learning rate | same as scalar baseline for first comparison |
| replay and target update | unchanged from baseline |

The first controlled experiment should alter only the scalar/distributional head and corresponding loss. Hyperparameter retuning should be reported separately from the architecture comparison.

---

## 9. Experimental Design

### 9.1 Core Configurations

| ID | Agent Head | Training Action Score | Mixer | Purpose |
|---|---|---|---|---|
| **A0** | Scalar | Expected Q | Soft-QMIX | Existing baseline |
| **A1** | IQN | Mean of quantiles | Soft-QMIX | Isolates distributional representation |
| **A2** | IQN | Fixed CVaR, `β=0.25` | Soft-QMIX | Tests strong risk sensitivity |
| **A3** | IQN | Annealed CVaR, `1.0→0.25` | Soft-QMIX | Full Phase-1 method |
| **A4** | IQN | Fixed CVaR, `β=0.50` | Soft-QMIX | Sensitivity to risk level |

The most important comparison is:

```text
A0 vs A1 vs A3
```

This separates the effect of learning a distribution from the additional effect of using the lower tail for decisions.

### 9.2 Evaluation Metrics

#### Standard Performance

- test battle win rate;
- mean episode return;
- median episode return;
- sample efficiency or steps required to reach fixed win-rate thresholds;
- final performance averaged over a fixed evaluation window;
- area under the learning curve.

#### Distributional and Robustness Metrics

- lower-quartile episode return;
- empirical `CVaR_0.25` of episode returns;
- 10th-percentile episode return;
- return variance and interquartile range;
- failure rate under difficult spawn configurations;
- win rate stratified by unit composition;
- seed-to-seed variance;
- calibration between predicted lower quantiles and realized returns.

#### Optimization Diagnostics

- quantile TD loss;
- mean predicted return;
- predicted quantile spread;
- average per-action CVaR;
- frequency of quantile crossing;
- gradient norm;
- target-online distribution discrepancy;
- policy entropy;
- effective number of available lower-tail samples.

### 9.3 Statistical Protocol

For each configuration and scenario:

- use the same pre-declared seed list;
- use identical environment seeds where feasible;
- report mean and median across seeds;
- report 95% bootstrap confidence intervals;
- compare final-window performance, not only peak performance;
- report learning curves with seed-level traces or shaded uncertainty;
- use paired tests when runs share matched environment seeds;
- report effect sizes in addition to p-values.

The paper should avoid declaring improvement from one unusually strong seed or from a transient peak.

---

## 10. Empirical Hypotheses

### H1 — Distributional Representation

The IQN model using mean-based action selection (`A1`) will equal or outperform the scalar baseline (`A0`) because it receives a richer Bellman training signal.

### H2 — Lower-Tail Robustness

The annealed-CVaR model (`A3`) will improve empirical lower-tail return and worst-quartile win rate more than it improves the mean.

### H3 — Difficult Episode Conditions

The improvement from `A3` over `A0` will be larger for unfavorable spawn and unit-composition strata than for easy strata.

### H4 — Risk Schedule

Annealing `β` from risk-neutral to risk-sensitive will train more reliably than using `β=0.25` from the beginning.

### H5 — Stability Across Seeds

The distributional method will reduce the frequency of late-training regression and lower variance across seeds, even when the mean win-rate gain is modest.

---

## 11. Theoretical Properties and Scope of Claims

### 11.1 Quantile-Wise Monotonicity

For a fixed sampled quantile `τ`, the existing mixer satisfies:

```text
∂Z_tot(s,a;τ) / ∂Z_i(τ_i,a_i;τ) ≥ 0
```

Therefore, for that quantile slice, increasing any local quantile utility cannot decrease the mixed joint quantile value.

This extends the baseline monotonicity constraint pointwise across sampled quantiles. It does not require a new mixer theorem because the mixer itself is unchanged.

### 11.2 Decentralized Action Selection

Both the mean and CVaR scores are computed independently from each agent's local return distribution. Therefore, decentralized execution is preserved:

```text
score_i(a_i) = E[Z_i]              or
score_i(a_i) = CVaR_β[Z_i]
```

Each agent can select its action without observing the global state during execution.

### 11.3 Risk Interpretation

CVaR is a lower-tail risk functional. For reward maximization, larger lower-tail CVaR means that an action has better outcomes among its least favorable predicted returns.

The paper should claim improved empirical risk sensitivity and robustness only if supported by lower-tail evaluation. It should not claim universal convergence or guaranteed policy improvement under nonlinear function approximation without a complete proof.

### 11.4 Quantile Validity

Quantile-wise mixing assumes that the learned samples meaningfully approximate ordered quantile functions. Because neural quantile models can exhibit quantile crossing, the implementation must log crossing frequency and may include a crossing penalty only as a separately reported ablation.

---

## 12. Expected Contribution to the Literature

The intended contribution is focused and testable:

> A distributional and risk-sensitive extension of GAT + Soft-QMIX for procedurally randomized cooperative multi-agent environments, evaluated on SMACv2 with explicit lower-tail robustness analysis.

The work is potentially valuable because much cooperative MARL evaluation emphasizes mean win rate while giving limited attention to the distribution of episode outcomes. SMACv2's randomized configurations provide a natural setting in which distributional critics and risk-sensitive decision rules can be evaluated.

The paper should position its novelty around the combination of:

- distributional per-agent utilities;
- quantile-wise entropy-regularized monotonic mixing;
- CVaR-based decentralized action selection;
- SMACv2 lower-tail and composition-conditioned evaluation.

The paper should not claim novelty for IQN, CVaR, GAT, QMIX, or distributional MARL individually.

---

## 13. Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| IQN increases training variance | Slower or unstable learning | Start with `K=8`, retain baseline optimizer, clip gradients, compare A1 before adding CVaR |
| Early CVaR policy is overly pessimistic | Insufficient exploration | Warm up with `β=1.0` and anneal gradually |
| Quantile-wise mixer produces crossing | Poor distribution interpretation | Log crossing rate; test sorting or a crossing penalty only as an ablation |
| Improvement appears only in tail metrics | Mean win rate may remain unchanged | Frame robustness as a primary outcome and report both mean and lower-tail results |
| Additional compute prevents enough seeds | Weak statistical evidence | Prioritize A0, A1, and A3; reduce secondary ablations before reducing seed count |
| Baseline and proposed method are not compute-matched | Unfair comparison | Use equal environment steps, update counts, evaluation frequency, and seed protocol |
| CVaR gains depend heavily on `β` | Fragile conclusion | Include at least `β∈{0.25,0.50,1.0}` or an annealed schedule |
| Distributional calibration is poor | Tail estimates may be misleading | Compare predicted quantiles with realized returns on held-out evaluation episodes |

---

## 14. Minimum Publishable Result

The minimum defensible paper requires:

1. a reproducible multi-seed scalar GAT + Soft-QMIX baseline;
2. a correctly implemented IQN distributional head;
3. an ablation separating distributional learning from CVaR action selection;
4. evaluation on more than mean win rate;
5. lower-tail robustness analysis;
6. confidence intervals and seed-level reporting;
7. honest reporting when the method improves robustness but not average performance.

A strong result would demonstrate that the full Phase-1 model improves both mean win rate and empirical lower-tail return. A still-meaningful result would show that mean performance is similar while difficult-condition and worst-quartile performance improve substantially.

---

## 15. Suggested Paper Structure

### 1. Introduction

- SMACv2 introduces randomized episode-level uncertainty.
- Scalar value decomposition optimizes expected return but discards return-distribution structure.
- Distributional modelling enables explicit lower-tail decision criteria.
- Summarize the method and contributions.

### 2. Related Work

- cooperative value decomposition;
- entropy-regularized QMIX variants;
- distributional reinforcement learning;
- distributional value factorization in MARL;
- risk-sensitive MARL;
- SMACv2 robustness and generalization.

### 3. Background

- Dec-POMDP and CTDE;
- QMIX/Soft-QMIX monotonic factorization;
- IQN and quantile regression;
- CVaR for reward maximization.

### 4. Method

- GAT baseline encoder;
- IQN per-agent distributional head;
- quantile-wise Soft-QMIX mixing;
- distributional Bellman objective;
- CVaR action selection;
- training and test policies.

### 5. Experimental Setup

- SMACv2 scenarios;
- baseline implementation;
- compute and hyperparameters;
- seed protocol;
- evaluation metrics;
- ablations.

### 6. Results

- learning curves;
- final-window win rate;
- lower-tail return and CVaR;
- spawn/composition stratification;
- risk-level sensitivity;
- calibration and quantile diagnostics.

### 7. Discussion

- whether gains arise from distributional representation or risk sensitivity;
- trade-off between mean and lower-tail performance;
- limitations of quantile-wise mixing;
- compute overhead;
- applicability beyond SMACv2.

### 8. Conclusion

- concise findings supported by the experiments;
- no discussion of unimplemented later phases as part of the contribution.

---

## 16. Reference Scaffold

The final paper should retain only references relevant to the Phase-1 contribution:

- Rashid et al., **QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning**.
- The baseline Soft-QMIX or entropy-regularized value-decomposition reference used by the current repository.
- Dabney et al. 2018, **Implicit Quantile Networks for Distributional Reinforcement Learning**.
- Sun et al. 2021, **DFAC: Distributional Value Function Factorization for Multi-Agent Reinforcement Learning**.
- Qiu et al. 2021, **RMIX: Learning Risk-Sensitive Policies for Cooperative Reinforcement Learning Agents**.
- Rockafellar and Uryasev 2000, **Optimization of Conditional Value-at-Risk**.
- Ellis et al., **SMACv2: An Improved Benchmark for Cooperative Multi-Agent Reinforcement Learning**.
- The GAT reference and the specific CASVD/GAT baseline references used by the implementation.
- Any current SMACv2 comparison method actually included in the experiments.

References related only to cellular sheaves, partial information decomposition, hyperscanning, future-conditioned role learning, and adaptive per-agent entropy are removed because those mechanisms are not part of the final method.

---

## Appendix A — Final One-Paragraph Description

We propose Risk-Aware Distributional Soft-QMIX, a focused extension of the existing GAT + Soft-QMIX cooperative MARL baseline for SMACv2. Instead of predicting one scalar utility for each agent action, the method learns an implicit quantile distribution over returns using an IQN-style head. The existing monotonic Soft-QMIX mixer is retained and applied independently across sampled quantiles, allowing the joint return distribution to be trained through quantile Huber regression without changing the underlying CTDE structure. During training, agents use a CVaR-based entropy-regularized policy that increasingly emphasizes the lower tail of predicted returns, with risk sensitivity annealed from mean-based behavior to a selected lower-tail level. The method is evaluated against the scalar baseline using multi-seed win rate, sample efficiency, empirical CVaR, worst-quartile return, and performance stratified by randomized spawn and unit composition. The paper's final contribution is limited to this distributional and risk-sensitive Phase-1 extension.

---

## Appendix B — Removed Scope

The following concepts from the earlier broad proposal are explicitly excluded from the final paper and implementation:

- cellular sheaf or sheaf-cochain mixing;
- learned restriction maps or sheaf Laplacians;
- partial information decomposition;
- synergy intrinsic rewards;
- future-conditioned role InfoNCE;
- multi-timescale synchrony losses;
- per-agent dual-descent entropy coefficients;
- a six-component SYNERGOS architecture;
- performance claims attributed to combinations of those components;
- implementation phases beyond the scalar baseline and distributional-CVaR extension.

They may be considered in separate future work, but they are not part of the present paper's method, experiments, or claimed contribution.
