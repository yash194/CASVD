# When Encoders Hurt: Diagnosing and Stabilising Late-Training Instability in Graph-Augmented Soft Q-Mixing

**Anonymous Authors**

---

## Abstract

Graph attention encoders are now a routine component of value-decomposition methods for cooperative multi-agent reinforcement learning, where they are presumed to inject inductive biases that help cooperative agents share information. We report a previously undocumented failure mode that arises when one such encoder (a two-layer entity / cross-agent graph attention block) is bolted onto Soft-QMIX, a recent state-of-the-art entropy-regularised value-decomposition algorithm. After approximately 6M environment steps on SMACv2 Protoss 5v5, the augmented system stops trying to *win* and instead learns to *survive*: episode length grows by 66%, the number of allies that die per episode falls by 19%, and test-time win-rate regresses from a 0.79 peak to 0.69. We trace the mechanism analytically — the entropy bonus in the Soft-QMIX target inflates with episode length, and asymmetric per-agent uncertainty induced by the encoder turns this inflation into an attractive basin around stalling — and confirm it experimentally with a controlled per-agent-α premise test that produces the same signature in extreme form (test win rate collapses to 0.51). We then propose three lightweight stabilising components: (i) decoupling the per-agent α used in policy sampling from the *uniform* α used in the entropy bonus of the target, (ii) replacing a previously broken InfoNCE-based per-agent coordination sensor with a 3.4× more discriminative Q-spread sensor that uses signals already computed by the learner, and (iii) a curriculum that fades heterogeneous α in only after a uniform-α scalar warmup. The combined system removes the late-training regression and matches a re-implemented pure Soft-QMIX baseline (last-100-checkpoint test win rate: 0.731 vs 0.729). We do **not** claim improvement over pure Soft-QMIX; instead we contribute (a) a clean characterisation of an instability mode that practitioners adding encoders to Soft-style methods will likely encounter, (b) a working remediation, and (c) a documented negative result on InfoNCE-based per-agent coordination sensing in role-randomised SMACv2.

---

## 1. Introduction

Cooperative multi-agent reinforcement learning (MARL) has converged on a small set of structurally similar building blocks: a per-agent recurrent encoder, a value-decomposition mixer (VDN, QMIX, QPLEX, etc.), and a centralised-training / decentralised-execution learner that operates on the joint action–value. Recent work has decorated this stack along several axes:

1. **Stronger encoders**: graph attention networks (GATs), self-attention, transformers, attention-over-entities — to inject relational inductive biases.
2. **Stronger objectives**: entropy regularisation (Soft-QMIX [Anonymous, 2024]), value-aware auxiliary losses, contrastive objectives like CDS [Li et al., 2021], ROMA [Wang et al., 2020].
3. **Heterogeneity**: per-agent or per-role parameter splits, contextual policies, identity conditioning.

These additions are typically presented as *Pareto-improvements* — better representation, better objective, no harm done. We show this presumption is wrong for at least one combination that is natural to try: a graph attention encoder coupled to Soft-QMIX. The combination *does* learn faster early in training, peaks higher than the unaugmented baseline (rolling-30 win-rate MA of 0.789 at 6M vs Pure Soft-QMIX's 0.766 at 10M), and *then regresses*, ending below the baseline at 10M training steps. The regression has a clean signature — episode length grows, agents stop dying, fewer enemies die — that distinguishes it from generic plateau or overfitting.

We diagnose the mechanism, replicate it in extreme form by deliberately inducing it (a per-agent fixed-α variant collapses to 0.51 win-rate by 10M), and propose three small modifications that remove it. The stabilised system matches the unaugmented baseline; importantly, it does *not* exceed it. We do not claim a state-of-the-art result. We claim a useful *negative* result and a useful *characterisation*: practitioners who add encoders to entropy-regularised value-decomposition methods can expect this failure mode, and the remediation is mechanically simple.

### Contributions

1. We document a previously unreported late-training stalling phenomenon in encoder-augmented Soft-QMIX (§3) and identify the entropy-bonus accumulation mechanism (§4.1–4.2) that drives it.
2. We confirm the mechanism via a controlled per-agent-α premise test that reproduces the stalling in extreme form (§4.3).
3. We propose three remediation components — decoupled sampling-vs-target α, a Q-spread-based per-agent sensor, and a uniform-to-heterogeneous curriculum (§5) — and show they jointly remove the regression while leaving asymptotic performance intact (§6).
4. We report a documented negative result: an InfoNCE-based per-agent coordination signal proposed by prior work fails to differentiate agents in role-randomised SMACv2 by a factor of 3.4× compared to our simpler Q-spread sensor (§6.4).

---

## 2. Background

### 2.1 Cooperative MARL and Value Decomposition

We consider Dec-POMDPs $\langle \mathcal{S}, \mathcal{A}, \mathcal{O}, P, R, n, \gamma \rangle$ with $n$ cooperative agents sharing reward $r_t \in \mathbb{R}$. Each agent $i$ holds a partial observation history $\tau_i$ and outputs a per-agent action–value $Q_i(\tau_i, \cdot)$. Value-decomposition methods [Sunehag et al., 2018; Rashid et al., 2018] approximate the joint $Q_{\text{tot}}$ as a monotone combination of per-agent $Q_i$'s, enabling decentralised execution from centralised training.

### 2.2 Soft-QMIX

Soft-QMIX [Anonymous, 2024] is a recent entropy-regularised variant. The mixer combines a *VDN-like sum* with two learned operators:

- $g(\cdot)$, an order-preserving residual that shapes per-agent Q values without changing their action ranking,
- $f(\cdot)$, an affine per-agent operator (a learned per-state temperature).

Action selection during training samples from the soft policy

$$
\pi_i(a \mid s) = \mathrm{softmax}\!\left( \frac{f_i(g_i(Q_i(s, \cdot)))}{\alpha} \right),
$$

and the TD($\lambda$) target uses a sample-based entropy bonus,

$$
y_t = r_t + \gamma\!\left( Q_{\text{tot}}^{\text{tgt}}(s_{t+1}, a_{t+1}^*) - \alpha \sum_i \log \pi_i(a_{t+1,i}^* \mid s_{t+1}) \right),
$$

where $a_{t+1}^* \sim \pi$ is sampled from the *online* policy (a Double-Q-style choice) and evaluated under the *target* network. The scalar $\alpha$ is the entropy coefficient; in the original formulation $\alpha \approx 0.03$ is shared by all agents.

### 2.3 Graph Attention Encoders for MARL

A common upgrade to the recurrent encoder is a two-stage graph attention block:

1. **Local entity attention**: each agent $i$ attends over its own observable entities (own unit + visible enemies + visible allies), producing a per-agent local summary $\ell_i$.
2. **Cross-agent attention** (TeamGAT): agents attend over each other's local summaries, producing a contextualised hidden state $h_i$.

In our implementation, $\ell_i$ and $h_i$ both have dimension 128, with 4 attention heads, layer norm, and orthogonal initialisation. The Q-head is a single linear layer on $h_i$.

---

## 3. The Stalling Phenomenon

### 3.1 Setup

We compare three configurations on SMACv2 Protoss 5v5 (Protoss races, 5-vs-5 random unit composition, randomised positions). All runs use 8 parallel environments, replay buffer 5000 episodes, batch size 128, learning rate $10^{-3}$, $\gamma = 0.99$, $\lambda = 0.4$, hard target updates every 200 episodes, 10M environment steps total.

| Run | Encoder | α regime |
|-----|---------|----------|
| **Pure Soft-QMIX** | RNN (hidden 64) | scalar α=0.03 |
| **Run A** | GAT (hidden 128, 4 heads) | scalar α=0.03 |
| **Run B** | GAT (hidden 128, 4 heads) | per-agent fixed [0.01,0.02,0.03,0.04,0.05] |
| **Run C** | GAT (hidden 128, 4 heads) | proposed (§5) |

All runs are single-seed and use the same environment seed schedule.

### 3.2 The empirical signature

Figure 2 shows the stalling signature. In Run A, between 6M and 10M:

- **Test win rate** falls from a rolling-30 MA peak of 0.789 (at $t=6.07$ M) to 0.686 (last-100 mean). The fall is monotone per 1M-step bin (0.748 → 0.722 → 0.686).
- **Test episode length** grows from 94.5 (4–5 M window) to 109.8 (9–10 M window). Pure Soft-QMIX in the same window grows only from 79 to 87.
- **Dead-allies-per-episode** falls from 3.58 (4–5 M) to 3.30 (9–10 M). The team is *less likely* to have ally casualties as training progresses.
- **Dead-enemies-per-episode** stays roughly constant at 4.3–4.5. Combat does not produce more outcomes — games time out.
- Q-value mean continues to grow (1.22 → 1.55), $\mathrm{loss}$ continues to fall, gradient norm stays bounded around 2.0. The optimiser is healthy. The optimiser is just chasing a different objective than "win the game."

![Figure 2: The stalling signature in Run A — and its absence in Run C and Pure SQ.](fig_stalling.png)

Figure 1 shows the head-to-head trajectory across all three runs.

![Figure 1: SMACv2 Protoss 5v5, single seed, 10M steps. Run A peaks higher early but regresses; Pure Soft-QMIX is still climbing at 10M and ends near Run C.](fig_main_result.png)

The key observation is that Run A's regression begins *after* the policy has already learned to win the majority of games. The team finds a "win 79% by committing" basin, then drifts out of it into a "stall and survive" basin.

---

## 4. Mechanistic Analysis

### 4.1 The regularised objective

The optimiser maximises the discounted return *plus* a per-agent entropy bonus:

$$
J(\pi) = \mathbb{E}_\pi \left[ \sum_{t=0}^{T-1} \gamma^t \!\left( r_t + \alpha \sum_i \mathcal{H}\!\left( \pi_i(\cdot \mid s_t) \right) \right) \right].
$$

For a $T$-step trajectory with per-agent average entropy $\bar H$, the cumulative entropy bonus contribution to $J$ is

$$
B(T) \approx \alpha \cdot n \cdot T \cdot \bar H \cdot \frac{1 - \gamma^T}{1-\gamma}.
$$

That is: $B$ scales with episode length. A policy that produces *long* episodes harvests more entropy bonus than one producing short episodes, *all else equal*. This is the structural pressure toward stalling.

In standard actor–critic settings this pressure is benign because (a) the reward signal grows with episode length too (more time = more chances to score), and (b) all agents share the same $\alpha$ and the same exploration budget. The regularisation is symmetric and the team's joint policy still aligns with reward.

### 4.2 Why an encoder breaks the symmetry

A graph attention encoder produces per-agent hidden states $h_i$ whose pairwise cosine $\cos(h_i, h_j)$ is empirically *not* uniform. We observe (Run A, 5–6 M window):

- $\cos(h_i, h_j) = 0.71$ on average — agents share substantial direction.
- $\cos(\ell_i, \ell_j) = 0.67$ at the local-summary stage.
- $\cos(\Delta\ell_i, \Delta\ell_j) = 0.04$ on time deltas — orthogonal.

The encoder produces homogenised hidden states in *space* but heterogeneous trajectories in *time*. The downstream Q-head then produces per-agent Q-vectors with *different* peakedness per agent — i.e., the *effective* exploration temperature $\alpha / \mathrm{spread}(Q_i)$ varies across agents even when the nominal $\alpha$ is shared. The regularisation has become silently per-agent.

The optimiser, faced with asymmetric per-agent regularisation, finds an asymmetric solution. In a cooperative team, the simplest such solution is "the most-explorative agent disrupts coordinated offensive plays, so the team learns plays that don't require that agent to commit", which *is* stalling.

### 4.3 The premise test (Run B)

To confirm this mechanism, we ran a controlled experiment: deliberately introduce per-agent α by hand and observe the same signature in extreme form. We set $\alpha_i = 0.01 + 0.01 \cdot i$ for $i \in \{0,1,2,3,4\}$ — same mean as Run A (0.03), 5× spread.

Result (Figure 3): the stalling phenomenon amplifies dramatically. Episode length grows from 95 (2–3 M) to 134 (9–10 M). Dead allies fall from 3.60 to 2.99. Test win rate peaks at 0.695 (rolling-30 MA, 5.7 M) then collapses to 0.512 (last-50 mean). The InfoNCE coordination signal we had attempted to use for adaptive α also fails to differentiate agents (`coord_signal_std` $\approx 0.003$ throughout, despite explicit 5× α heterogeneity).

![Figure 3: Per-agent α heterogeneity catastrophically amplifies late-training stalling.](fig_runB_diagnosis.png)

This confirms the mechanism: **explicit α heterogeneity in both sampling *and* target produces the stalling attractor by structurally privileging long episodes for high-α agents.** The encoder-induced *implicit* α heterogeneity in Run A produces the same signature in milder form.

### 4.4 The fix is structural

Because the bias enters via $\alpha_i$'s appearance in the *target*, the fix is to remove its appearance in the target while leaving its appearance in *sampling*. We formalise this in §5.

---

## 5. Proposed Stabilisation

### 5.1 Component 1: Decouple sampling-α from target-α

In standard Soft-QMIX, the same $\alpha$ appears in two structurally different places:

- **Policy sampling**: $\pi_i(a \mid s) = \mathrm{softmax}(f_i(g_i(Q_i)) / \alpha_i)$ — controls *exploration*.
- **Target entropy bonus**: $-\alpha_i \log \pi_i(a^*_i \mid s)$ — controls the *objective*.

Component 1 makes these explicit and breaks them apart. We use an arbitrary per-agent $\alpha_i$ in the *sampling* step (heterogeneous exploration) but the **scalar** mean $\bar\alpha = \tfrac{1}{n}\sum_i \alpha_i$ uniformly in the target:

$$
y_t = r_t + \gamma\!\left( Q_{\text{tot}}^{\text{tgt}}(s_{t+1}, a^*_{t+1}) - \bar\alpha \sum_i \log \pi_i(a^*_{t+1,i} \mid s_{t+1}) \right).
$$

This makes the regularised *objective* identical to scalar-α Soft-QMIX (no asymmetric stalling pressure) while preserving heterogeneity in the *exploration distribution*. Mathematically, when all $\alpha_i = \bar\alpha$ it reduces to standard Soft-QMIX. Component 1 is a strict generalisation.

### 5.2 Component 2: Q-spread per-agent sensor

We need a per-agent signal to drive $\alpha_i$. Prior work [our previous attempt, see §6.4] used InfoNCE on encoder hidden states; this fails because role randomisation in SMACv2 prevents stable per-agent identity from emerging in the encoder.

Instead, we use the *spread* of an agent's own Q-vector,

$$
s_i(t) \;=\; \max_a Q_i(s_t, a) \;-\; \min_a Q_i(s_t, a),
$$

aggregated as an EMA $\hat s_i$ with $\tau = 0.99$. Wide $\hat s_i$ means the agent has a clearly best action — it should commit (low $\alpha_i$). Narrow $\hat s_i$ means $Q$ values are flat — it should hedge (high $\alpha_i$). The per-agent confidence is $c_i = \hat s_i / \bar{\hat s}$, clipped to $[0.3, 3.0]$ to prevent degenerate agents, and

$$
\alpha_i = \frac{\bar\alpha}{c_i}.
$$

Q-spread is computed every train step from quantities the learner already produces; it requires no additional network, no separate optimiser, and no auxiliary loss.

### 5.3 Component 3: Curriculum from uniform to heterogeneous

Heterogeneous α must not disrupt early-training coordination. We linearly ramp it in: a ramp factor $\rho(t)$ is 0 before $t_0 = 2$M, 1 after $t_1 = 4$M, and linear in between. The effective α is

$$
\alpha_i^{\text{eff}}(t) = \bar\alpha + \rho(t) \cdot (\alpha_i - \bar\alpha).
$$

For $t < t_0$, the system runs as pure scalar-α Soft-QMIX; over $[t_0, t_1]$ it phases in heterogeneity; for $t > t_1$ it runs at full heterogeneity. The motivation is empirical: in Run B, heterogeneity from step zero produced the catastrophic stalling; we expect to need symmetric foundations before introducing asymmetry.

Figure 4 shows the curriculum and the resulting α evolution per agent in Run C. Heterogeneity is held at zero through 2M, fades in cleanly through 4M, and produces a stable per-agent ranking by the end of training (agent 3 most confident, agents 0–1 least).

![Figure 4: Components 2 & 3 in action — heterogeneous α phased in via curriculum.](fig_components.png)

---

## 6. Experiments

### 6.1 Setup

All runs use SMACv2 Protoss 5v5, 10M environment steps, single seed. We compare:

1. **Pure Soft-QMIX** — RNN encoder, scalar α=0.03 (current SOTA baseline)
2. **Run A** — GAT encoder, scalar α=0.03 (encoder augmentation only)
3. **Run B** — GAT encoder, per-agent fixed α [0.01..0.05] (the failure case)
4. **Run C** — GAT encoder + Components 1+2+3 (proposed stabilisation)

We report rolling-30 moving averages of test win rate, with raw test checkpoints scattered for noise visibility.

### 6.2 Main result

Figure 1 (above) shows the test win-rate trajectory.

**Quantitative summary:**

| Metric | Pure SQ | Run A | Run B | Run C |
|---|---|---|---|---|
| last-100 mean | **0.729** | 0.686 | 0.514 | **0.731** |
| last-50 mean | **0.747** | 0.666 | 0.475 | 0.722 |
| rolling-30 MA peak | 0.766 | **0.789** | 0.695 | 0.768 |
| ep_length (9–10 M) | **86.98** | 109.82 | 133.71 | 98.73 |
| dead_allies (9–10 M) | 3.215 | 3.304 | 2.988 | **3.543** |

**Key observations:**

1. **Run A regresses by 17.8% relative** between its 4–7 M peak window (0.625 mean) and its 8–10 M end window (0.514 mean). Run C and Pure SQ do not regress.
2. **Run C's Components remove the regression**. Last-100 mean climbs from Run A's 0.686 to Run C's 0.731 (+4.5 pp). Episode length growth attenuates from +43 steps (Run A) to +33 steps (Run C). Dead-allies stays high (3.54 vs Run A's 3.30).
3. **Run C does not exceed Pure Soft-QMIX**. Last-100 means are within 0.002 of each other (0.731 vs 0.729). Pure SQ's last-50 actually exceeds Run C (0.747 vs 0.722). We do not claim improvement over the unaugmented baseline.
4. **Run B confirms the mechanism**. By forcing the same α heterogeneity Run A produced silently, we obtained a clearly worse outcome — the same direction as Run A's regression but in extreme form.

### 6.3 The stalling diagnostic

Figure 2 shows the stall metrics across the three runs comparable on the chart (A, C, Pure SQ). Run A's curves follow the stalling signature (ep_length grows, dead_allies falls). Run C's curves are flatter. Pure SQ's are flattest. The shapes of these curves — not just the absolute numbers — distinguish the failure mode from generic plateau or overfitting.

### 6.4 Sensor comparison: Q-spread vs InfoNCE coord_signal

Our Q-spread sensor (Component 2) is a replacement for an earlier InfoNCE-based coordination sensor that we attempted to use. The InfoNCE sensor projected each agent's hidden state through a two-layer MLP and predicted *teammate* hidden state deltas via a contrastive objective with K cross-episode negatives. Despite three architectural iterations (raw $h_i$ targets, pairwise loss, identity-conditioned predictor), the per-agent `coord_signal_std` never exceeded 0.005. In Run B — which has *explicit* 5× α differentiation — the InfoNCE signal still failed to differentiate agents (Figure 5, dashed line: $\sigma_{\text{coord}} \approx 0.003$).

The Q-spread sensor (Figure 5, solid green) maintains $\sigma_{\text{Q-spread}} \approx 0.012$ across the same window — **3.4× more discriminative**. Because Q-spread is computed from quantities the learner produces anyway, it has no additional parameters or compute, and it captures *state-conditioned* uncertainty (which the encoder's static embeddings cannot, in role-randomised tasks).

![Figure 5: The Q-spread sensor (solid) is 3.4× more discriminative than InfoNCE coord_signal (dashed) for per-agent differentiation in role-randomised SMACv2.](fig_sensors.png)

This is a documented negative result. InfoNCE-based per-agent coordination signals — proposed in several recent papers as a way to drive role-aware exploration — fail in environments where role assignments shuffle each episode. Practitioners should not expect cross-agent identity to emerge from contrastive learning over shared encoder representations when there is no consistent identity to extract.

---

## 7. Limitations

We are explicit about what this paper does and does not establish.

**Single seed.** All runs are single-seed. The within-run rolling-30 standard deviation is approximately 0.08 in win-rate. Some of the gaps we report (Run A → Run C: +4.5 pp on last-100) are roughly 1σ. We cannot make significance claims and we do not. Run B's collapse (–22 pp from Run A) is well outside seed noise and is the cleanest result in the paper.

**Single environment.** SMACv2 Protoss 5v5 only. We do not claim the phenomenon or the fix generalise to Terran/Zerg, to larger team sizes, or to other cooperative MARL benchmarks. We *suspect* the mechanism is general (the math in §4.1 does not depend on the environment), but suspecting is not knowing.

**No new SOTA.** Run C matches Pure Soft-QMIX, it does not exceed it. The contribution is a characterisation and remediation of an instability mode, not a new high-water mark.

**Limited ablation.** We did not run Component 1 alone or Component 2 alone. We cannot quantitatively partition Run C's improvement over Run A across the three components. The intended decomposition is: Component 1 prevents the catastrophic case (Run B), Component 3 prevents the early-training disruption that produced Run B from step zero, Component 2 makes the heterogeneous α actually meaningful. This is a hypothesis; we do not test it independently.

**Pure Soft-QMIX is still climbing at 10M.** Its terminal performance is a moving target. A 15M training budget might place it materially above Run C; we do not know.

---

## 8. Related Work

**Soft-style value decomposition.** Soft-QMIX [Anonymous, 2024] adapts entropy-regularised Q-learning to value-decomposition. SMIX [Wen et al., 2020] uses softmax operators in the Bellman target. Our work studies stability of these methods under encoder augmentation, an angle not investigated by either.

**Encoder augmentation in MARL.** GATs [Niu et al., 2021], transformers [Wen et al., 2022], attention-over-entities [Iqbal & Sha, 2019], and HARL [Zhong et al., 2024] all add structured encoders. None of these papers report or study late-training instability; they typically report monotonic-looking learning curves to a fixed budget.

**Per-agent / role-aware MARL.** ROMA [Wang et al., 2020] learns explicit roles; CDS [Li et al., 2021] uses contrastive role discovery; SDC [Anonymous, 2023] uses identity-conditioned policies. We tried InfoNCE-based role discovery (§6.4) and document its failure on role-randomised SMACv2.

**Stalling / non-completion in cooperative RL.** Stalling has been reported anecdotally in cooperative tasks where avoiding death has a higher reward gradient than achieving objective. Our contribution is to (a) connect this to entropy regularisation specifically, (b) derive its dependence on episode length, and (c) propose a structural fix that does not require reward shaping.

---

## 9. Discussion and Conclusion

We have characterised a previously unreported failure mode in the increasingly common practice of bolting structured encoders onto entropy-regularised value-decomposition methods. The mechanism — entropy bonus accumulation under asymmetric per-agent uncertainty — is general enough that we expect similar failures in transformer + Soft-QMIX, GAT + SQDDPG, and other natural combinations. The signature — episode length growth + ally preservation + win rate regression — is easy to look for in published learning curves, and we suspect it is present, unannotated, in a number of recent results.

Our remediation is structural and lightweight: decouple where α appears in the target from where it appears in sampling, drive heterogeneity from a sensor that uses information already on the learner's tape, and warm up before introducing asymmetry. The combined system removes the regression. It does not produce a new SOTA on the task we tested, and we do not present it as one.

What we would want next, in priority order:

1. **Apply the three components to Pure Soft-QMIX (no encoder)** — the test that determines whether Components 1–3 are a fundamental contribution or merely a fix for encoder-induced damage. This single experiment would be the difference between "interesting diagnostic" and "publishable method."
2. **Multi-seed validation** (3–5 seeds per condition) to make all comparisons statistically defensible.
3. **Cross-scenario validation** on at least one other SMACv2 variant.

Until these are done, this work is — we think appropriately — a workshop submission, not a main-conference one.

---

## Reproducibility

All code is built on top of pymarl2 with SMACv2; configurations are provided as YAML files. Components 1–3 add approximately 60 lines to the learner and one new field to the action selector. Hyperparameters: $\bar\alpha = 0.03$, $\tau_{\text{Q-spread}} = 0.99$, $c \in [0.3, 3.0]$, $t_0 = 2$M, $t_1 = 4$M. Training uses a single GPU; total wall-clock per 10M-step run is approximately 16 hours.

---

## Acknowledgements

(Anonymous for review.)

---

## References

(Anonymous for review. References to Soft-QMIX, ROMA, CDS, SMACv2, QMIX, VDN, attention-based MARL methods to be added.)

