# CASVD: Coordination-Aware Soft Value Decomposition

## A Novel Algorithm for Cooperative Multi-Agent Reinforcement Learning

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Background: Why Existing Approaches Fail](#background-why-existing-approaches-fail)
   - [Why Vanilla SAC Fails in Cooperative MARL](#why-vanilla-sac-fails-in-cooperative-marl)
   - [Why QMIX Has Limitations Too](#why-qmix-has-limitations-too)
3. [Diagnosis of Current GAT-SAC Implementation](#diagnosis-of-current-gat-sac-implementation)
   - [The Damning Numbers](#the-damning-numbers)
   - [Critical Issue 1: Alpha Divergence](#critical-issue-1-alpha-is-diverging--target-entropy-is-unreachable)
   - [Critical Issue 2: Q-Value Overestimation and Critic Instability](#critical-issue-2-q-value-overestimation--critic-instability)
   - [Critical Issue 3: No Parameter Sharing Between Agents](#critical-issue-3-no-parameter-sharing-between-agents)
   - [Major Issue 4: LGDD Pre-training is Counterproductive](#major-issue-4-lgdd-pre-training-is-counterproductive)
   - [Major Issue 5: SAC Fundamentally Conflicts with Cooperative MARL](#major-issue-5-sac-fundamentally-conflicts-with-cooperative-marl)
   - [Major Issue 6: Critic Architecture is Mismatched](#major-issue-6-critic-architecture-is-mismatched)
   - [Moderate Issues 7-10](#moderate-issues)
4. [Understanding the Current Data Flow](#understanding-the-current-data-flow)
   - [Where GAT Embeddings Go](#where-gat-embeddings-go)
   - [Why the Critic Does Not Use GAT Embeddings](#why-the-critic-does-not-use-gat-embeddings)
   - [Why This Flow is Suboptimal](#why-this-flow-is-suboptimal)
5. [The CASVD Algorithm: Complete Design](#the-casvd-algorithm-complete-design)
   - [Core Insight: Coordination Should Control Exploration](#core-insight-coordination-should-control-exploration)
   - [Math 1: Soft Value Decomposition](#math-1-soft-value-decomposition)
   - [Math 2: Coordination-Aware Alpha](#math-2-coordination-aware-alpha-the-novel-part)
   - [Math 3: Cross-Agent Dynamics Prediction (LGDD v2)](#math-3-cross-agent-dynamics-prediction-lgdd-v2)
   - [Math 4: No Separate Actor Network](#math-4-no-separate-actor-network)
6. [Full Architecture Diagram](#full-architecture-diagram)
7. [Training Algorithm (Pseudocode)](#training-algorithm-pseudocode)
8. [Component Role Mapping](#component-role-mapping)
9. [Why CASVD Outperforms Each Baseline](#why-casvd-outperforms-each-baseline)
   - [vs QMIX](#vs-qmix-and-wqmix-qplex)
   - [vs SAC / MASAC](#vs-sac--masac)
   - [vs MAPPO](#vs-mappo)
   - [vs UPDeT](#vs-updet-transformer--qmix)
10. [The CL Component (Now Justified)](#the-cl-component-now-justified)
11. [Implementation Roadmap](#implementation-roadmap)
12. [Hyperparameters](#hyperparameters-starting-points)
13. [NeurIPS Paper Framing](#neurips-paper-framing)
14. [Appendix: Detailed Issue Explanations](#appendix-detailed-issue-explanations)

---

## Executive Summary

CASVD (Coordination-Aware Soft Value Decomposition) is a novel multi-agent reinforcement learning algorithm that unifies the coordination guarantee of value decomposition (QMIX) with the adaptive exploration of entropy-regularized Q-learning (SAC). The key innovation is using a cross-agent dynamics prediction objective (LGDD v2) whose prediction error serves as an intrinsic coordination signal that automatically modulates the entropy temperature:

- When agents cannot predict each other's dynamics → high entropy → explore
- When agents can predict each other → low entropy → exploit coordination

This creates a single algorithm that avoids both relative overgeneralization (QMIX's weakness) and coordination destruction (SAC's weakness), while uniquely enabling zero-shot transfer across team compositions via physics-grounded entity representations.

---

## Background: Why Existing Approaches Fail

### Why Vanilla SAC Fails in Cooperative MARL

Standard SAC was designed for single-agent continuous control. When applied to cooperative multi-agent tasks, it fails for five fundamental reasons:

#### Problem 1: Entropy Maximization Destroys Coordination

SAC's core objective for each agent:

```
maximize:  E[Q(s,a)] + α * H(π)
                         ↑
              "be as random as possible"
```

Winning in StarCraft requires focus fire — all agents attacking one enemy:

```
Good strategy: All 5 agents attack enemy_2 (who is low HP)
Probability under coordinated policy: high
Probability under maximum-entropy policy: (1/5)^5 = 0.00032

SAC actively PUNISHES this coordination because it's "low entropy"
```

Each agent independently tries to be random. The probability that 5 independently random agents accidentally coordinate is exponentially small. SAC doesn't just fail to find coordination — it actively pushes away from it once found, because coordinated play is deterministic (low entropy) and SAC penalizes that.

QMIX has no such problem. Epsilon-greedy exploration decays to 0.05. Once good joint actions are found, agents exploit them. Nothing pushes them back toward randomness.

#### Problem 2: Independent Policy Optimization ≠ Joint Optimization

SAC optimizes each agent's policy independently:

```
Agent 0: maximize E[Q_0(s, a_0)] + α * H(π_0)
Agent 1: maximize E[Q_1(s, a_1)] + α * H(π_1)
Agent 2: maximize E[Q_2(s, a_2)] + α * H(π_2)
...
```

Each agent greedily picks the action that looks best for itself. But cooperative MARL has a joint reward — what matters is the joint action (a_0, a_1, a_2, a_3, a_4), not individual actions.

Classic failure case:

```
Enemy_1 has 10 HP.  One attack does 8 damage.
Enemy_2 has 10 HP.

Optimal joint strategy: 2 agents attack enemy_1 (kills it), 3 attack enemy_2

What SAC does:
  Agent 0's critic: Q(attack_enemy_1) = 5.2, Q(attack_enemy_2) = 5.1
  Agent 1's critic: Q(attack_enemy_1) = 5.2, Q(attack_enemy_2) = 5.1
  ...
  All 5 agents attack enemy_1 → overkill (wasted 3 attacks)
  OR entropy pushes them to spread randomly → no kill at all
```

The per-agent critic can't express "agent 0 should attack enemy_1 IF agent 1 also attacks enemy_1." It only sees individual Q-values.

QMIX solves this structurally:

```
Q_total = mixer(Q_1, Q_2, Q_3, Q_4, Q_5, state)

With monotonicity: ∂Q_total/∂Q_i ≥ 0

This guarantees: if agent_i improves its local Q → joint Q improves
The mixer learns to weight agents so that the JOINT outcome is optimized
```

#### Problem 3: The Critic Can't Capture Multi-Agent Credit Assignment

The SAC critic outputs Q(s, a_i) for each agent independently. But the reward is shared — all agents get the same team reward.

```
Episode: Team kills enemy_3. Reward = +10 for everyone.

Who actually contributed?
- Agent 0 dealt the killing blow
- Agent 1 was kiting, drawing fire (crucial but no damage)
- Agent 2 was already dead
- Agent 3 was attacking enemy_5 (wasted effort)
- Agent 4 was healing agent 0 (enabled the kill)

SAC critic sees: reward = +10 for ALL agents
  → Agent 2 (dead, did nothing): "my Q-value should be +10" ← WRONG
  → Agent 3 (wrong target): "my Q-value should be +10" ← WRONG
  → Agent 1 (kiting): "my Q-value should be +10" ← correct but for wrong reason
```

The critic has no mechanism to assign credit. Every agent gets the same reward regardless of individual contribution. Over many episodes, the Q-values converge to the average team reward — not a useful signal for individual action selection.

QMIX's value decomposition IS credit assignment:

```
Q_total = mixer(Q_1, Q_2, Q_3, Q_4, Q_5)

TD error on Q_total → gradient flows through mixer → each Q_i gets
a DIFFERENT gradient based on how much it contributed to Q_total.

The mixer learns: "Agent 0's action mattered a lot this step,
Agent 2's action didn't matter (dead)."
```

#### Problem 4: Soft Q-Values Are Unstable in Multi-Agent Settings

In single-agent SAC, the entropy bonus adds a smooth, well-behaved term. One agent's policy changes slowly → value estimates are stable.

In multi-agent SAC with shared reward, each agent's environment is NON-STATIONARY because other agents' policies change:

```
Step 1000:  Agent 1 always attacks enemy_1 → Agent 0 learns to attack enemy_2
Step 2000:  Agent 1 switches to enemy_3 → Agent 0's Q-values are now WRONG
Step 3000:  Agent 0 adapts → Agent 1's Q-values are now WRONG
...forever
```

SAC makes this worse than epsilon-greedy because:
- Epsilon-greedy: policies change slowly (epsilon decays, argmax is stable)
- SAC entropy: policies are FORCED to keep changing (must maintain randomness)
- More policy variation → more non-stationarity → more unstable Q-values

#### Problem 5: Alpha Tuning Has No Multi-Agent Solution

In single-agent SAC, target entropy = -dim(A) (continuous) or log(n_actions) * 0.98 (discrete). This works because there's one agent with a fixed action space.

In cooperative MARL:
- Effective action space varies (dead agents have 1 action, living agents have ~6)
- The "right" entropy depends on what teammates are doing
- Agents that need to coordinate should have LOW entropy (agree on targets)
- Agents that need to explore should have HIGH entropy
- A single α for all agents at all times cannot capture this

There's no principled way to set target entropy in multi-agent SAC. Every value is wrong for some agents in some states.

### Why QMIX Has Limitations Too

QMIX is not perfect either. Its key weakness is **relative overgeneralization**:

```
Strategy A: Both agents attack together → reward +10 (but if one defects → -5)
Strategy B: Both agents play safe     → reward +3 (guaranteed)

QMIX + epsilon-greedy:
  Agent 0 tries Strategy A, Agent 1 (random) doesn't cooperate → reward -5
  Agent 0 learns: "Strategy A is bad" → converges to Strategy B forever
  Probability of BOTH trying A simultaneously: 0.05 * 0.05 = 0.0025
  QMIX gets stuck in the mediocre Strategy B.
```

This is where SAC's entropy COULD help — if properly controlled, it would keep exploring Strategy A long enough for both agents to discover it simultaneously.

### Summary: SAC vs Value Decomposition for Cooperative MARL

| Aspect | SAC | Value Decomposition (QMIX) |
|---|---|---|
| Coordination | Actively punished by entropy | Guaranteed by monotonic mixer |
| Credit assignment | None (shared reward → same Q for all) | Mixer gradient differentiates contributions |
| Joint optimization | Each agent optimizes independently | Joint Q optimized, decomposed to agents |
| Non-stationarity | Amplified by entropy forcing variation | Reduced by stable epsilon-greedy |
| Exploration | Entropy (exponentially unlikely to find joint strategies) | Epsilon-greedy (decays, allows exploitation) |
| Relative overgeneralization | Resists it (entropy keeps exploring) | Vulnerable (converges to safe strategy) |
| Tuning | α, target_entropy — no principled multi-agent solution | ε schedule — simple, well-understood |
| Sample efficiency | Good (off-policy) | Good (off-policy) |

Neither is complete. CASVD combines both.

---

## Diagnosis of Current GAT-SAC Implementation

### The Damning Numbers

Training on protoss_5_vs_5 at ~950k-995k timesteps (about halfway through 2.05M total):

| Metric | Value | What It Means |
|---|---|---|
| `battle_won_mean` | **0.000** | Zero training wins at 950k steps |
| `test_battle_won_mean` | **0.033** | 3% test win rate — near random |
| `dead_allies_mean` | **5.000** | ALL 5 allies die every episode |
| `dead_enemies_mean` | **0.39** | Barely scratching the enemy |
| `alpha` | **2.89 and RISING** | Entropy weight exploding |
| `Q_mean` | **~28** | Q-values 5x higher than actual returns (~5.5) |
| `critic_grad_norm` | **125-523** | 12x-52x above clip threshold (10) |
| `entropy` | **~1.17** | Below unreachable target of 2.35 |
| `lgdd_loss` | **0.037-0.042** | Small and stagnant |
| `lgdd_grad_norm` | **0.007-0.01** | Effectively zero contribution |
| `cl_distill_loss` | **0.0004-0.0007** | Teacher ≈ student (both bad) |
| `return_mean` | **~5.5** | Very low returns |
| `ep_length_mean` | **~55** | Episodes end quickly (limit ~200) |

### Critical Issue 1: Alpha is Diverging — Target Entropy is Unreachable

This is the single biggest problem killing the algorithm.

```python
# sac_learner.py:86
self.target_entropy = -th.log(th.tensor(1.0 / self.n_actions)).item() * 0.98
```

For protoss_5_vs_5: `n_actions = 6 + max(5,5) = 11`, so `target_entropy = log(11) * 0.98 ≈ 2.35`.

But this target is unreachable for three reasons:

**1. Action masking kills achievable entropy.** Dead agents can ONLY no-op (entropy = 0). Living agents can only attack visible enemies + move. In practice, 4-6 actions are available, not 11. Maximum achievable entropy ≈ log(5) ≈ 1.6.

**2. Optimal play requires coordination, not randomness.** A good policy SHOULD be more deterministic — focus fire on one enemy, kite melee units, etc. Target entropy of 2.35 forces near-uniform randomness.

**3. The alpha update loop is a death spiral:**

```
entropy (1.17) < target (2.35) → alpha increases → policy pushed toward randomness
→ random policy can't coordinate → agents die → more dead agents (entropy=0)
→ average entropy drops further → alpha increases MORE
```

The alpha loss formula is:

```python
alpha_loss = (self.log_alpha * (avg_entropy - self.target_entropy).detach())
```

This formula is mathematically correct (the sign is right — it should NOT be negated). The problem is the target value, not the formula.

When entropy (1.17) < target (2.35): gradient is negative → SGD increases log_alpha → alpha increases. Since the target can never be reached, alpha increases forever. At alpha = 2.89 and still rising, the entropy bonus overwhelms meaningful Q-value differences.

### Critical Issue 2: Q-Value Overestimation + Critic Instability

Q-values (~28) are ~5x the actual returns (~5.5). The targets (~28) confirm the bootstrapping is propagating this error.

**Root cause A: Alpha inflates soft Q-values.** The target computation:

```python
target_v = (target_probs * (target_q_min - self.alpha * target_log_probs)).sum(dim=-1)
```

With alpha=2.89, each step adds ~2.89 * entropy to the value, inflating Q-values above actual returns. As alpha rises, Q-values rise, but they become disconnected from reward signal.

**Root cause B: SACTypeCritic's additive structure amplifies overestimation.** For 3 unit types:

```python
# sac_type_critic.py:137
q1_out[:, :, agent_idx, :] = q1_coord + q1_type
```

The GlobalCoordinator adds a bonus on top of the type-specific Q. Both can independently overestimate, and the sum amplifies the error. The twin-Q min trick (`min(Q1, Q2)`) helps for each network pair, but the additive structure creates a bias that min doesn't correct.

**Root cause C: Critic gradient norms of 125-523 with clip at 10.** This means 90-98% of gradient information is destroyed every update. The critic is barely learning. The direction is preserved but magnitude is crushed, making updates ineffective for correcting large TD errors.

### Critical Issue 3: No Parameter Sharing Between Agents

```python
# gat_ns_agent.py:148
self.agents = nn.ModuleList([
    IndividualAgentNet(self.hidden_dim, self.n_actions, ...)
    for _ in range(self.n_agents)
])
```

5 separate RNN + policy heads, zero weight sharing. Each agent's policy head sees only its own data — 1/5 the effective sample size.

#### Information Sharing ≠ Parameter Sharing

**Information sharing** (what TeamGAT does) = agents can see each other at runtime. TeamGAT lets agent 0 know what agents 1-4 are doing. That's communication. It answers: "What are my teammates doing right now?"

**Parameter sharing** (what's missing) = agents use the same neural network weights during training. This answers: "How should ANY agent behave given this situation?"

These solve completely different problems. TeamGAT provides communication (good, keep it). But the lack of parameter sharing means:

**With separate networks (current):**
- Episode 1: Agent 0 = Stalker, Agent 2 = Zealot
- Episode 2: Agent 0 = Zealot, Agent 2 = Stalker (procedural generation shuffled them)
- Agent 0's network was trained on Stalker data, now must handle Zealot
- Agent 2's network was trained on Zealot data, now must handle Stalker
- Each network gets a random mix of unit types, learning none well

**With a shared network (fix):**
- ONE network processes ALL 5 agents' experiences every episode
- It sees 5x more data per training step
- It learns: "when I'm a Stalker (indicated by input features), do X. When I'm a Zealot, do Y."
- The unit type is in the observation already (`obs_agent_id: True` + unit_type_bits)

QMIX uses `n_rnn` with full parameter sharing — every agent's experience trains the same network. This is a major reason QMIX learns faster.

#### The Credit Assignment Nightmare

Separate networks also create a credit assignment problem:

```
Episode: all 5 agents die, reward = 0

With shared network:
  → gradient says "this BEHAVIOR (given these features) was bad"
  → all 5 agents' experiences contribute to the same update
  → learns fast what works and what doesn't

With 5 separate networks:
  → agent 0's network: "I (as a random unit type) did badly"
  → agent 3's network: "I (as a different random unit type) did badly"
  → each network gets 1/5 the signal, mixed across unit types
  → none of them learn which behavior was actually bad
```

### Major Issue 4: LGDD Pre-training is Counterproductive

#### The Chicken-and-Egg Problem That Doesn't Exist

The motivation for LGDD pre-training was to solve a chicken-and-egg problem: SAC needs good GAT embeddings to train, and GAT needs good SAC gradients to learn. But this problem doesn't actually exist in the current architecture.

The critic takes RAW state + RAW obs as input — it does NOT use GAT embeddings:

```python
# sac_type_critic.py:147-158
pieces = [
    batch["state"],     # RAW global state
    batch["obs"],       # RAW observation
    agent_id_onehot     # agent identity
]
```

So the training flow works fine from step 0:
1. Critic sees raw (state, obs) → learns Q-values from rewards (works fine, no GAT dependency)
2. Actor (GAT) outputs random policy → collects experience (fine, every RL algorithm starts random)
3. Critic has slightly better Q-values → Actor loss adjusts policy → GAT starts learning

There is no deadlock. The critic bootstraps independently on raw inputs, then its Q-values guide the actor. This is how every actor-critic algorithm works.

#### What LGDD Actually Does (Counterproductively)

During Phase 1 (0-100k steps):
- Random actions are taken (`lgdd_random_warmup: True`)
- Encoder is trained via LGDD to predict next-state latents from random actions
- RL gradients are blocked from encoder (`detach_encoder=True`)

**Problem 1: Random actions produce chaotic dynamics.** Predicting z_{t+1} from random a_t teaches the encoder about what happens when agents act randomly — not what happens under coordinated play. The "physics" learned is trivially basic (things near each other interact), not the useful physics (focus fire dynamics, kiting, ability interactions).

**Problem 2: 100k steps of zero RL learning.** That's ~5% of the 2.05M budget spent learning useless representations.

**Problem 3: Phase transition at 100k steps.** When RL gradients suddenly flow at step 100k, the encoder has representations tuned for dynamics prediction from random actions, not Q-value estimation. LGDD wants features that capture position/velocity/health trends. RL wants features that distinguish "attack enemy 3" vs "attack enemy 4". These are different objectives requiring different features.

During Phase 2 (100k+ steps):
- LGDD continues with weight 0.1
- `lgdd_grad_norm: 0.007-0.01` — effectively zero contribution
- LGDD is doing nothing but adding compute cost

### Major Issue 5: SAC Fundamentally Conflicts with Cooperative MARL

See the detailed analysis in [Background: Why Vanilla SAC Fails](#why-vanilla-sac-fails-in-cooperative-marl) above. The core conflicts are:

1. Entropy maximization actively punishes coordination
2. Independent optimization doesn't optimize joint actions
3. No credit assignment mechanism for shared rewards
4. Non-stationarity amplified by entropy-driven policy variation
5. No principled way to set target entropy for multi-agent settings

### Major Issue 6: Critic Architecture is Mismatched

The actor (GAT) and critic (MLP) have a massive capability gap:

```
Actor (GAT):    Raw obs → entity encoders → multi-head attention → team GAT → RNN → policy
                ↑ understands entities, relationships, team structure

Critic (MLP):   Raw state + raw obs → Linear → Linear → Q-values
                ↑ sees a flat vector of numbers, no structure
```

The critic is the teacher. The actor can only be as good as the Q-values guiding it. A 2-layer MLP with 128 hidden units cannot disentangle entity relationships from a flat ~245-dimensional vector the way attention can.

In CTDE (Centralized Training, Decentralized Execution), the critic should be the MOST powerful component — it's only used during training, so computational cost doesn't matter. The current design has it backwards: the actor is complex, the critic is simple.

With 3 unit types, the critic has 3 TwinQNetworks (6 Q-networks) + 2 GlobalCoordinators = 8 networks. This is massively over-parameterized for 5 agents, yet all 8 networks share the same MLP architecture that can't do attention over entities.

### Moderate Issues

**Issue 7: Reward standardization hides signal.** With 0% wins, the reward distribution is very narrow (small damage values). Standardization compresses this further. When the agent finally does something good, the reward looks similar to other steps.

**Issue 8: Batch size 32 is too small for SAC.** QMIX uses 128. With stochastic policies and per-agent networks, larger batches are needed to reduce gradient variance.

**Issue 9: `gain: 0.01` makes critic learn slowly.** All weights initialized near zero means Q-values start near zero and take many updates to reach meaningful values.

**Issue 10: CL is premature.** Continual learning prevents forgetting, but there's nothing worth remembering. The `cl_distill_loss ≈ 0.0007` confirms teacher ≈ student (both equally bad). It's just adding overhead.

### Why QMIX Works Better Despite Being "Simpler"

| Aspect | QMIX | GAT-SAC |
|---|---|---|
| Parameter sharing | Full (1 network for all agents) | None (5 separate networks) |
| Exploration | Epsilon-greedy (1.0→0.05) | SAC entropy (forces randomness forever) |
| Batch size | 128 | 32 |
| Learning rate | 0.001 | 0.0003 |
| Critic | Mixing network with monotonicity constraint | 8 unconstrained MLPs |
| Coordination | Built-in via value decomposition | Hope agents learn to coordinate |
| Sample efficiency | High (shared params, large batch) | Low (5x less per-agent data) |
| Complexity | Low (~3 components) | Very high (~8+ interacting components) |

QMIX's value decomposition guarantees that improving each agent's local Q-value improves the joint Q-value. GAT-SAC has no such guarantee.

---

## Understanding the Current Data Flow

### Where GAT Embeddings Go

```
                         ACTOR (policy network)
                         ══════════════════════
Raw obs ──→ split into [move | enemy | ally | own]
                              │
                    encode into node embeddings
                              │
                     LocalEntityGAT (attention)
                              │
                      local_summary (z_t)
                              │
                     TeamGATLayer (attention)
                              │
                      team_summary (g_t)          ← THIS is the GAT embedding
                              │
                     IndividualAgentNet (RNN + Linear)
                              │
                        action logits
                              │
                     softmax + mask
                              │
                    action probabilities (π)  ──→  used to PICK actions in env
                              │
                         log_probs (log π)


                         CRITIC (Q network)
                         ══════════════════
Raw state + Raw obs + agent_id  ──→  MLP  ──→  Q(s, a) for all actions
         (no GAT involved)
```

### Why the Critic Does Not Use GAT Embeddings

The critic and actor are completely separate networks connected only through the loss function:

```python
# Actor loss (sac_learner.py:199)
actor_loss = (probs * (alpha * log_probs - q_pi_min)).sum(dim=-1)
#              ↑              ↑               ↑
#          from GAT       from GAT      from critic (no GAT)
```

The Q-values GUIDE the GAT, but gradients don't flow into the critic. The critic just provides the target signal:

```
actor_loss
    │
    ├──→ d(loss)/d(probs) → d(probs)/d(logits) → d(logits)/d(GAT weights)
    │    ↑ gradients flow through the actor (GAT encoder)
    │
    └──→ Q values: NO gradient (detached with th.no_grad())
         ↑ critic just provides a number, no gradient connection
```

GAT embeddings are used for:

| Where | How |
|-------|-----|
| Action selection (rollout) | GAT → logits → probs → sample action |
| Actor loss (training) | GAT → probs, log_probs → compared against Q-values |
| LGDD (auxiliary) | GAT → z_t, g_t → predict z_{t+1}, g_{t+1} |
| CL distillation | student GAT probs vs teacher GAT probs |

### Why This Flow is Suboptimal

The critic (teacher) is dumber than the actor (student):

```
Smart actor (GAT) ←── guided by ──── Dumb critic (MLP)

Dumb critic: "attack enemy 2 has Q=28.1, attack enemy 3 has Q=27.9"
These numbers are NOISE — the MLP can't distinguish the two
Smart actor tries to learn from noise → learns nothing useful
```

If the critic also had GAT, a chicken-and-egg problem would arise (critic depends on GAT, actor depends on critic). If the critic had its own separate GAT, that doubles parameters wastefully.

The solution: **eliminate the actor-critic split entirely with value decomposition.** The GAT encoder produces Q-values directly, and the mixer handles coordination. No separate critic needed. No chicken-and-egg possible.

---

## The CASVD Algorithm: Complete Design

### Core Insight: Coordination Should Control Exploration

The fundamental problem: QMIX always exploits (epsilon-greedy) and SAC always explores (entropy maximization). Neither adapts.

```
QMIX:  Always greedy → gets stuck in suboptimal equilibria
SAC:   Always random → can't coordinate
```

What we need: exploration when agents are uncoordinated, exploitation when they've found coordination. But how do you measure coordination?

**Answer: LGDD prediction error.**

When agent i can predict what the team will do next (low LGDD error) → agents are behaving predictably toward each other → they are coordinated → reduce entropy → allow deterministic play.

When agent i cannot predict the team's next state (high LGDD error) → agents are unpredictable → uncoordinated → increase entropy → search for coordination.

This repurposes LGDD from a useless pre-training trick into the algorithm's coordination sensor.

### Math 1: Soft Value Decomposition

Standard QMIX decomposes Q-values with monotonicity:

```
Q_total(s, a) = mixer(Q_1(o_1, a_1), ..., Q_n(o_n, a_n), state)
with ∂Q_total/∂Q_i ≥ 0
```

We extend this to soft Q-values with per-agent entropy:

**Policy** (derived from Q-values, no separate actor network):

```
π_i(a | o_i) = exp(Q_i(o_i, a) / α_i) / Z_i

where Z_i = Σ_a exp(Q_i(o_i, a) / α_i)  (partition function)
```

**Soft value** per agent (used in bootstrap target):

```
V_i^soft(o_i) = α_i * log Σ_a exp(Q_i(o_i, a) / α_i)
```

This is the log-sum-exp operation, which smoothly interpolates:
- When α_i → 0: V_i^soft → max_a Q_i (becomes QMIX-style greedy)
- When α_i → ∞: V_i^soft → mean_a Q_i (becomes uniform, maximum exploration)

**So α_i smoothly interpolates between QMIX and maximum-entropy SAC.**

**TD target:**

```
y = r + γ * mixer_target(V_1^soft(o_1'), ..., V_n^soft(o_n'), state')
```

**TD loss:**

```
L_critic = (Q_total(s, a) - y)²

where Q_total = mixer(Q_1(o_1, a_1), ..., Q_n(o_n, a_n), state)
```

Gradients flow end-to-end: L → mixer → each Q_i → GRU → TeamGAT → LocalEntityGAT → entity encoders. One loss. No actor-critic split.

### Math 2: Coordination-Aware Alpha (The Novel Part)

Instead of fixed or globally-tuned alpha, α is a function of the coordination signal:

```
α(t) = α_min + (α_max - α_min) * σ(β * (ε_coord(t) - ε̄_coord))

where:
  ε_coord(t)  = LGDD cross-agent prediction error (smoothed via EMA)
  ε̄_coord     = running mean of ε_coord (adaptive threshold)
  β           = sensitivity hyperparameter (~5.0)
  σ           = sigmoid function
  α_min       = 0.01  (nearly greedy when coordinated)
  α_max       = 1.0   (exploratory when uncoordinated)
```

**The dynamics create a virtuous cycle:**

```
Training starts: random policy → LGDD error HIGH → α ≈ α_max → explore freely
Agents begin coordinating: LGDD error DROPS → α decreases → allow exploitation
Agents get stuck in suboptimal equilibrium: LGDD error STAGNATES
  → α stays moderate → entropy prevents full convergence on bad strategy
Agents find better strategy: LGDD error drops FURTHER → α drops more → commit
New team composition (SMACv2 procedural gen): LGDD error SPIKES
  → α jumps up → explore the new composition → find coordination → α drops again
```

This is self-calibrating — no hand-tuned target entropy, no diverging alpha, no death spiral.

### Math 3: Cross-Agent Dynamics Prediction (LGDD v2)

Current LGDD: agent i predicts its OWN next latents.

```
Current: (z_i, g_i, a_i) → predict z_i_{t+1}, g_i_{t+1}
Problem: Measures "is the environment predictable", not "are agents coordinated"
```

LGDD v2: agent i predicts the TEAM's next state.

```
New: (z_i, g_i, a_i) → predict g_{t+1} (team-level next state)

This measures: "Can agent i predict what the TEAM will do next?"
If yes → agent i understands the coordination pattern → coordinated
If no  → agent i is out of sync → needs exploration
```

The prediction error becomes the coordination signal:

```
ε_coord(t) = EMA(||pred_g_{t+1} - target_g_{t+1}||², τ=0.99)

target from EMA encoder (slowly updated copy, stable targets)
```

Loss (auxiliary, co-trained with TD loss from step 0):

```
L_lgdd = (1/T) Σ_t ||predictor(z_t, g_t, a_t) - sg(target_encoder(obs_{t+1}))_team||²

where sg = stop gradient (targets don't receive gradients)
```

Key differences from current LGDD:
1. No random warmup phase — LGDD runs alongside RL from step 0
2. No detach_encoder — RL and LGDD both train the encoder simultaneously
3. Predicts team summary (cross-agent), not just own summary
4. Prediction error drives alpha, not just representation learning

### Math 4: No Separate Actor Network

In discrete action spaces, the policy can be derived directly from Q-values (Boltzmann/softmax):

```
π_i(a | o_i) = softmax(Q_i(o_i, ·) / α_i)
```

This eliminates:
- The actor-critic mismatch (dumb critic, smart actor)
- The chicken-and-egg problem entirely
- Half the parameters and half the optimizers
- The need for the SACTypeCritic

The GAT encoder IS both the actor (through Boltzmann policy) and the individual Q-function (through the mixer, it becomes the critic).

```
Training: GAT → Q_i → mixer → Q_total → TD loss trains Q_i
Execution: GAT → Q_i → argmax(Q_i) or Boltzmann(Q_i / α)
```

---

## Full Architecture Diagram

```
                    ┌─────────────────────────────────────────────┐
                    │          SHARED GAT ENCODER                 │
                    │                                             │
  obs_i ──────────→ │  split → [move|enemy|ally|own]              │
                    │    │                                        │
                    │  encode → entity node embeddings            │
                    │    │                                        │
                    │  LocalEntityGAT (multi-head attention)      │
                    │    │                                        │
                    │  local_summary z_i ──────────────────────→ LGDD v2
                    │    │                                        │  │
                    │  TeamGATLayer (inter-agent attention)        │  │
                    │    │                                        │  │
                    │  team_summary g_i ──────────────────────→ LGDD v2
                    │    │                                        │  │
                    │  Shared GRU (one network, all agents)       │  ↓
                    │    │                                        │ ε_coord
                    │  Q_i(a) for all actions                     │  │
                    └────┬────────────────────────────────────────┘  │
                         │                                           │
         ┌───────────────┼───────────────────────────┐              │
         ↓               ↓               ↓           ↓              │
     Q_1(a_1)       Q_2(a_2)       Q_3(a_3)    Q_n(a_n)           │
         │               │               │           │              │
         └───────┬───────┴───────┬───────┘           │              │
                 ↓               ↓                                  │
         ┌──────────────────────────────────┐                      │
         │     QMIX MIXER (hypernetwork)    │                      │
         │                                  │                      │
         │  Q_total = f(Q_1,...,Q_n, state) │                      │
         │  ∂Q_total/∂Q_i ≥ 0              │                      │
         └──────────────┬───────────────────┘                      │
                        │                                           │
                    TD Loss                                         │
                        │                                           │
                        │        ┌──────────────────────────┐      │
                        │        │   ALPHA MODULE            │      │
                        │        │                          │←─────┘
                        │        │  α = α_min + Δα*σ(β*(ε-ε̄))    │
                        │        │                          │
                        │        └────────┬─────────────────┘
                        │                 │
                        │                 ↓
                        │         Policy (for action selection):
                        │         π_i(a) = softmax(Q_i(a) / α)
                        │
                    TOTAL LOSS = L_td + λ * L_lgdd
```

---

## Training Algorithm (Pseudocode)

```python
def train(batch, t_env):

    # ──────────────────────────────────────
    # 1. Forward pass through shared GAT encoder
    # ──────────────────────────────────────
    init_hidden(batch.batch_size)
    all_q_values = []
    all_z = []      # local summaries for LGDD
    all_g = []      # team summaries for LGDD

    for t in range(batch.max_seq_length):
        inputs_t = build_inputs(batch, t)    # obs + agent_id
        z_t, g_t = gat_encoder.encode(inputs_t)
        q_t = shared_q_head(gat_encoder.gru(g_t, hidden))
        all_q_values.append(q_t)
        all_z.append(z_t)
        all_g.append(g_t)

    q_values = stack(all_q_values, dim=1)
    # shape: (batch, T, n_agents, n_actions)

    # ──────────────────────────────────────
    # 2. Compute Q_total via QMIX mixer
    # ──────────────────────────────────────
    actions = batch["actions"][:, :-1]
    q_taken = gather(q_values[:, :-1], dim=3, index=actions).squeeze(3)
    # q_taken shape: (batch, T-1, n_agents)

    q_total = mixer(q_taken, batch["state"][:, :-1])
    # q_total shape: (batch, T-1, 1)

    # ──────────────────────────────────────
    # 3. Compute coordination-aware alpha
    # ──────────────────────────────────────
    lgdd_errors = []
    for t in range(batch.max_seq_length - 1):
        with no_grad():
            target_g_next = target_encoder.encode(
                build_inputs(batch, t + 1)
            )["team_summary"]

        pred_g_next = dynamics_predictor(
            all_z[t], all_g[t], batch["actions_onehot"][:, t]
        )["pred_team"]

        lgdd_error_t = mse(pred_g_next, target_g_next, reduction="none").mean(dim=-1)
        lgdd_errors.append(lgdd_error_t)

    lgdd_errors = stack(lgdd_errors, dim=1)
    current_coord_error = (lgdd_errors * mask).sum() / mask.sum()

    # EMA update of coordination error baseline
    coord_error_ema = coord_ema_tau * coord_error_ema + (1 - coord_ema_tau) * current_coord_error.item()

    # Coordination-aware alpha
    alpha = alpha_min + (alpha_max - alpha_min) * sigmoid(
        alpha_beta * (current_coord_error.item() - coord_error_ema)
    )

    # ──────────────────────────────────────
    # 4. Compute soft target values
    # ──────────────────────────────────────
    with no_grad():
        target_q = target_q_network(batch)  # (batch, T, n_agents, n_actions)

        # Mask unavailable actions
        avail_next = batch["avail_actions"][:, 1:]
        target_q_next = target_q[:, 1:]
        target_q_next[avail_next == 0] = -1e10

        # Per-agent soft value: V_i = α * logsumexp(Q_i / α)
        v_soft = alpha * logsumexp(target_q_next / alpha, dim=-1)
        # shape: (batch, T-1, n_agents)

        # Mix soft values through target mixer
        target_q_total = target_mixer(v_soft, batch["state"][:, 1:])

        # TD target
        rewards = batch["reward"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        targets = rewards + gamma * (1 - terminated) * target_q_total

    # ──────────────────────────────────────
    # 5. TD loss
    # ──────────────────────────────────────
    td_error = (q_total - targets.detach())
    mask_total = mask[:, :, 0:1]  # (batch, T-1, 1) — one value per timestep
    L_td = ((td_error ** 2) * mask_total).sum() / mask_total.sum()

    # ──────────────────────────────────────
    # 6. LGDD auxiliary loss (co-trained, no phases)
    # ──────────────────────────────────────
    L_lgdd = (lgdd_errors * mask[:, :, 0:1].expand_as(lgdd_errors)).sum() / mask.sum()

    # ──────────────────────────────────────
    # 7. Total loss and update
    # ──────────────────────────────────────
    total_loss = L_td + lgdd_weight * L_lgdd

    optimizer.zero_grad()
    total_loss.backward()
    clip_grad_norm(all_params, max_norm=10)
    optimizer.step()

    # ──────────────────────────────────────
    # 8. Soft target updates
    # ──────────────────────────────────────
    soft_update(target_network, online_network, tau=0.005)
    soft_update(target_mixer, mixer, tau=0.005)
    soft_update(target_encoder, gat_encoder, tau=0.01)

    # ──────────────────────────────────────
    # 9. CL: update teacher and apply distillation (if enabled)
    # ──────────────────────────────────────
    if cl_enabled and memory_batch is not None:
        soft_update(teacher_network, online_network, tau=0.002)
        L_cl = compute_cl_distillation(memory_batch, teacher_network)
        # Add to next training step or apply separately
```

---

## Component Role Mapping

How every component from the original GAT-SAC gets a precise, justified role in CASVD:

| Component | Old Role (Broken) | New Role (Justified) |
|---|---|---|
| **LocalEntityGAT** | Actor encoder (disconnected from critic) | Shared encoder for Q-values, feeds both policy AND mixer |
| **TeamGATLayer** | Actor communication (critic couldn't see it) | Team context for Q-values, AND coordination signal source |
| **LGDD** | Pre-training trick (random actions, then useless) | **Coordination sensor** — prediction error drives α |
| **Alpha (entropy)** | Fixed target, diverges forever | **Coordination-adaptive** — auto-tunes via LGDD error |
| **Type critic** | 8 separate MLPs (over-parameterized) | Replaced: shared Q-head handles types via input features |
| **CL (reservoir buffer)** | Premature (nothing to remember) | **Composition memory** — maintains cross-composition LGDD accuracy |
| **CL (teacher distill)** | Useless (teacher = student) | **Anti-forgetting** — preserves strategies for seen compositions |
| **Parameter sharing** | None (5 separate agents) | Full sharing through single GRU + Q-head |
| **Mixer** | Didn't exist | QMIX mixer provides coordination guarantee |
| **Separate actor** | Smart actor guided by dumb critic | Eliminated: policy derived from Q-values directly |
| **Separate critic** | Dumb MLP that can't reason about entities | Eliminated: mixer + Q_i IS the critic |

---

## Why CASVD Outperforms Each Baseline

### vs QMIX (and WQMIX, QPLEX)

```
Scenario: Relative overgeneralization

  Strategy A: Both agents attack together → reward +10
              (but if one defects → -5)
  Strategy B: Both agents play safe → reward +3 (guaranteed)

  QMIX: epsilon=0.05 → agents converge to B → stuck forever
        Probability of BOTH trying A: 0.05 * 0.05 = 0.0025

  CASVD: early training, LGDD error high → α ≈ 1.0 → agents exploratory
         Both try A with reasonable probability → discover +10 reward
         LGDD error drops → α drops → commit to A

  CASVD escapes the trap. QMIX doesn't.
```

### vs SAC / MASAC

```
Scenario: Focus fire coordination

  MASAC: entropy pushes all agents to be random → can't focus fire
         α keeps rising because target entropy unreachable
         Even with centralized critic, entropy prevents coordination

  CASVD: as agents learn to focus fire → LGDD error drops → α drops to α_min
         α_min = 0.01 → policy becomes nearly deterministic
         Value decomposition GUARANTEES individual improvements help team

  CASVD allows coordination. SAC/MASAC fight it.
```

### vs MAPPO

```
Scenario: Sample efficiency

  MAPPO: on-policy → uses each sample once → needs ~10M+ steps
  CASVD: off-policy (replay buffer) → reuses samples → needs ~2M steps

  MAPPO: clipped surrogate → conservative updates
  CASVD: TD learning → can take larger steps when confident (low α)
```

### vs UPDeT (Transformer + QMIX)

```
Scenario: Compositional generalization (5v5 → 10v10)

  UPDeT: Transformer encoder + QMIX. No dynamics prediction.
         Learns representations for seen compositions only.
         Transfers poorly — representations are task-specific.

  CASVD: GAT encoder + LGDD dynamics prediction + QMIX.
         LGDD forces representations to capture entity PHYSICS.
         Physics transfers: "stalker shoots from range" is true in 5v5 or 10v10.

         On new composition: LGDD error spikes → α increases → explores
         → quickly finds coordination → α drops → exploits

  CASVD transfers. UPDeT doesn't (or transfers much less).
```

---

## The CL Component (Now Justified)

CL is no longer premature. It has a precise role in handling SMACv2's procedural generation:

```
Episode 1: Team = [Stalker, Stalker, Zealot, Zealot, Colossus]
  → LGDD learns to predict dynamics for this composition
  → Q-network learns good strategy

Episode 100: Team = [Zealot, Zealot, Zealot, Stalker, Stalker]
  → LGDD error spikes (new dynamics!) → α increases → explore
  → Q-network needs to adapt

Without CL: adapting to new composition overwrites old composition knowledge
With CL:
  - Reservoir buffer replays old compositions
  - Teacher distillation preserves learned strategies
  - LGDD maintains cross-composition prediction ability
  - Net effect: learns ALL compositions, doesn't forget any
```

CL + LGDD create a synergy:
- LGDD detects composition change (error spikes) → triggers exploration
- CL prevents forgetting of previously learned compositions
- Together: fast adaptation + no catastrophic forgetting

---

## Implementation Roadmap

### Phase 1: Core CASVD (Replace SAC with Soft Value Decomposition)

**Goal:** Get a working value decomposition with soft Q-values and parameter sharing.

```
Files to create:
  1. src/learners/casvd_learner.py — TD loss with soft values through mixer
  2. src/controllers/casvd_controller.py — derive policy from Q via Boltzmann
  3. src/config/algs/casvd.yaml — new hyperparameters

Files to modify:
  1. src/modules/agents/gat_ns_agent.py — replace nn.ModuleList with shared Q-head
  2. src/learners/__init__.py — register casvd_learner
  3. src/controllers/__init__.py — register casvd_controller

Files leveraged as-is:
  1. src/modules/mixers/qmix.py — QMIX mixer (already in repo)

Files NO LONGER needed:
  - src/modules/critics/sac_type_critic.py (replaced by mixer)

Test: should learn on protoss_5_vs_5 with fixed alpha=0.5
Expected: 30-50% win rate within 1M steps
```

### Phase 2: Coordination-Aware Alpha via LGDD

**Goal:** Make alpha adaptive based on LGDD prediction error.

```
Files to modify:
  1. src/learners/casvd_learner.py — add LGDD error tracking, alpha computation
  2. src/modules/predictors/lgdd_predictor.py — modify to predict team summary
  3. src/config/algs/casvd.yaml — alpha_min, alpha_max, beta parameters

Test: compare fixed-alpha vs coordination-aware alpha
Expected: coordination-aware reaches higher win rate AND adapts faster
```

### Phase 3: Continual Learning for Compositional Generalization

**Goal:** Add CL replay and teacher distillation, justified by LGDD coordination signal.

```
Files to modify:
  1. src/learners/casvd_learner.py — add CL replay and teacher distillation
  2. src/run/run.py — reservoir buffer sampling (already exists)

Test: train on protoss_5_vs_5, test zero-shot on protoss_10_vs_10
Expected: CASVD transfers, baselines don't
```

### Phase 4: Ablations and Paper Experiments

```
Run configurations:
  1. Full CASVD (everything)
  2. CASVD - LGDD (fixed alpha, no dynamics prediction)
  3. CASVD - soft values (pure QMIX, no entropy)
  4. CASVD - GAT (flat MLP encoder)
  5. CASVD - CL (no replay/distillation)
  6. CASVD - cross-agent prediction (self-prediction only)

Baselines:
  7. QMIX
  8. WQMIX
  9. QPLEX
  10. MAPPO
  11. UPDeT + QMIX

Environments:
  - protoss_5_vs_5 (standard benchmark)
  - terran_5_vs_5 (different race)
  - protoss_10_vs_10 (scale transfer)
  - train protoss → test terran (cross-type transfer)
  - 20_vs_20 variants (large scale)
```

---

## Hyperparameters (Starting Points)

```yaml
# ─── Algorithm identity ───
name: "casvd"
learner: "casvd_learner"
mac: "casvd_mac"
agent: "gat_ns"          # reuse GAT encoder with shared Q-head modification
mixer: "qmix"

# ─── QMIX mixer ───
mixing_embed_dim: 32
hypernet_embed: 64

# ─── Shared GAT encoder ───
hidden_dim: 128
n_heads: 4
use_rnn: True
use_layer_norm: True
use_orthogonal: True
gain: 0.01

# ─── Soft value decomposition ───
alpha_min: 0.01           # nearly greedy when coordinated
alpha_max: 1.0            # exploratory when uncoordinated
alpha_beta: 5.0           # sensitivity of alpha to LGDD error
coord_ema_tau: 0.99       # smoothing for coordination error baseline

# ─── LGDD v2 (coordination sensor) ───
lgdd_enabled: True
lgdd_weight: 0.2          # auxiliary loss weight (constant, no phases)
lgdd_lr: 0.0003
lgdd_ema_tau: 0.01        # target encoder EMA update rate
lgdd_predictor_hidden_dim: 128
# NO random warmup, NO pretrain phase, NO detach_encoder

# ─── Training ───
lr: 0.0005
batch_size: 128
buffer_size: 5000
gamma: 0.99
grad_norm_clip: 10
target_update_interval_or_tau: 0.005
optimizer: "adam"
optimizer_epsilon: 0.0000001

# ─── Runner ───
runner: "parallel"
batch_size_run: 8

# ─── Observation ───
obs_agent_id: True
obs_last_action: False

# ─── CL (for transfer experiments) ───
cl_enabled: True
cl_memory_buffer_size: 10000
cl_current_ratio: 0.8
cl_min_memory_episodes: 128
cl_teacher_ema_tau: 0.002
cl_distill_weight: 0.05

# ─── Reward ───
standardise_rewards: False    # let raw reward signal through
standardise_returns: False
```

---

## NeurIPS Paper Framing

### Title

"Coordination-Aware Soft Value Decomposition with Self-Predictive Entity Representations for Cooperative Multi-Agent Reinforcement Learning"

Or shorter: "CASVD: Dynamics-Guided Coordination in Soft Value Decomposition"

### Abstract (Draft)

We introduce CASVD, which unifies value decomposition and entropy-regularized Q-learning for cooperative multi-agent RL. The key insight is that a self-predictive dynamics objective (LGDD), which trains entity-structured graph attention representations to predict future team states, simultaneously provides (1) a transferable representation learning signal and (2) an intrinsic coordination metric. When agents can predict each other's dynamics, they are coordinated — and the entropy temperature automatically decreases, allowing deterministic joint actions like focus fire. When dynamics become unpredictable (new team composition, suboptimal equilibrium), entropy increases, driving exploration. This creates a single algorithm that avoids both relative overgeneralization (QMIX's weakness) and coordination destruction (SAC's weakness), while uniquely enabling zero-shot transfer across team compositions via physics-grounded representations. We evaluate on SMACv2 with procedurally generated team compositions and demonstrate superior performance over QMIX, WQMIX, QPLEX, MAPPO, and UPDeT, with unique zero-shot transfer capabilities to unseen team sizes and compositions.

### Key Claims

1. We propose soft value decomposition: combining QMIX-style monotonic mixing with entropy-regularized soft Q-values, smoothly interpolating between greedy and exploratory regimes via a single temperature parameter α.

2. We introduce coordination-aware entropy: using cross-agent dynamics prediction error as an intrinsic coordination signal to automatically modulate α — high prediction error triggers exploration, low error enables exploitation.

3. We use entity-level graph attention to produce structured representations that generalize across varying team compositions in SMACv2.

4. Our method outperforms value decomposition methods (which lack exploration) and SAC-based methods (which lack coordination guarantees) on challenging SMACv2 scenarios, and uniquely transfers to unseen team compositions.

### Contributions

1. **Soft Value Decomposition** — A principled combination of value decomposition and entropy-regularized Q-learning for discrete cooperative MARL, with formal monotonicity guarantees preserved.

2. **Coordination-Aware Entropy** — A novel mechanism that uses self-predictive dynamics error as an intrinsic signal to automatically tune the exploration-exploitation tradeoff, eliminating the need for hand-tuned entropy targets.

3. **Cross-Agent Dynamics Prediction** — Repurposing latent dynamics prediction from a representation learning auxiliary to a coordination measurement tool, showing that prediction error is a reliable proxy for coordination quality.

4. **Compositional Generalization** — Demonstrating that entity-structured representations trained with dynamics prediction transfer zero-shot to unseen team compositions and sizes.

### Ablations (Critical for Reviewers)

| Ablation | What it removes | Expected effect | What it proves |
|---|---|---|---|
| CASVD - LGDD | No dynamics prediction, fixed α | Worse on hard tasks, no transfer | Dynamics prediction + adaptive α matter |
| CASVD - soft values | No entropy, pure QMIX | Gets stuck in suboptimal equilibria | Entropy helps escape local minima |
| CASVD - coordination-aware α | Static α throughout training | Either too much exploration or too little | Adaptive α is necessary |
| CASVD - GAT | Flat MLP encoder | Can't generalize across compositions | Entity structure matters |
| CASVD - cross-agent prediction | Self-prediction only | Weaker coordination signal | Cross-agent prediction is better metric |
| CASVD - parameter sharing | Separate per-agent networks | Much worse sample efficiency | Sharing is critical |
| CASVD - CL | No replay/distillation | Forgets old compositions | CL matters for generalization |
| CASVD - mixer | No value decomposition | Can't coordinate (like SAC) | Decomposition is essential |

### The Killer Experiment

```
Train on protoss_5_vs_5 (random Stalker/Zealot/Colossus compositions)

Test zero-shot on:
  - protoss_10_vs_10 (scale transfer)
  - terran_5_vs_5 (cross-race transfer)
  - protoss_20_vs_23 (asymmetric scale)

Expected results:
  QMIX:      60% win on 5v5, 10% on 10v10 transfer
  MAPPO:     55% win on 5v5, 5% on 10v10 transfer
  UPDeT:     65% win on 5v5, 20% on 10v10 transfer
  CASVD:     70% win on 5v5, 45% on 10v10 transfer  ← significantly better

The gap on TRANSFER is the headline result.
```

---

## Appendix: Detailed Issue Explanations

### A1: Why the Alpha Loss Sign is Correct

The alpha loss in the current implementation:

```python
alpha_loss = (self.log_alpha * (avg_entropy - self.target_entropy).detach())
```

This is mathematically correct. Tracing the math:

```
gradient w.r.t. log_α = (H - H_target)

When H (1.17) < H_target (2.35):
  gradient = -1.18 (negative)
  SGD update: log_α -= lr * (-1.18)  →  log_α INCREASES  →  α INCREASES
  Correct: entropy is too low, increase α to encourage exploration

When H > H_target:
  gradient is positive
  SGD update: log_α -= lr * (positive)  →  log_α DECREASES  →  α DECREASES
  Correct: entropy is too high, decrease α
```

Adding a negative sign would invert this behavior and be wrong. The bug is the unreachable target value (2.35), not the formula sign.

### A2: How Parameter Sharing Works with Different Unit Types

With separate networks, each agent has its own private brain:

```python
# Current (broken)
self.agents = nn.ModuleList([
    IndividualAgentNet(...)  # agent 0's private brain
    IndividualAgentNet(...)  # agent 1's private brain
    ...
])
```

With parameter sharing, one shared brain processes all agents:

```python
# Fixed
self.shared_agent = IndividualAgentNet(hidden_dim, n_actions, use_rnn=True)

# Forward: process all agents through the SAME network
all_inputs = team_summary.reshape(bs * n_agents, hidden_dim)
all_hidden = hidden.reshape(bs * n_agents, hidden_dim)
all_logits, all_next_hidden = self.shared_agent(all_inputs, all_hidden)
logits = all_logits.reshape(bs, n_agents, n_actions)
```

The shared network differentiates agent behavior through:
- `obs_agent_id: True` — one-hot agent identity in the input
- `unit_type_bits` — unit type information in the observation
- Different entity features (each agent sees different allies/enemies based on position)

Same TeamGAT communication. Same entity attention. Just one brain instead of five, conditioned on who it is via the input features.

### A3: The Complete Data Flow in Current GAT-SAC vs CASVD

**Current GAT-SAC flow (broken):**

```
obs → GAT encoder → team_summary → 5 separate policy heads → action probs
                                                                    │
                      ┌─────────────────────────────────────────────┘
                      ↓
              action probs compared against
                      ↓
raw state + obs → MLP critic → Q-values (no entity structure)
                      ↓
              actor loss = probs * (α * log_probs - Q)

Problems: critic is dumb, no coordination, α diverges, no sharing
```

**CASVD flow (fixed):**

```
obs → GAT encoder → team_summary → shared GRU → Q_i(a) per agent
                         │                            │
                         ↓                            ↓
                    LGDD predictor              QMIX mixer
                         │                            │
                         ↓                            ↓
                    ε_coord → α(s)              Q_total
                                                      │
                                                      ↓
                                                  TD loss

Policy: π_i(a) = softmax(Q_i(a) / α)  (derived from Q, no separate actor)

Everything trained end-to-end by one TD loss + LGDD auxiliary.
```

### A4: Modified SAC Can Work But Requires Specific Fixes

Standard SAC fails in cooperative MARL, but modified versions (MASAC, FACMAC) can be effective if three specific changes are made:

1. **True Centralized Critic**: The critic must see ALL agents' actions, not just individual observations. This eliminates non-stationarity.

2. **Entropy Scheduling**: Instead of fixed target entropy, decay it over training so agents can eventually converge on coordinated deterministic play.

3. **Counterfactual Credit Assignment**: Use counterfactual baselines (COMA-style) so each agent knows how much IT contributed vs the team average.

CASVD achieves all three goals more elegantly:
- Value decomposition replaces the centralized critic
- LGDD-driven alpha replaces entropy scheduling
- Monotonic mixing provides natural credit assignment through gradient decomposition

### A5: Why LGDD v2 (Cross-Agent Prediction) is Better Than Original LGDD

**Original LGDD:**
- Agent predicts its OWN next state: (z_i, g_i, a_i) → z_i_{t+1}, g_i_{t+1}
- Measures: "Is the environment predictable?"
- Problem: Environment can be predictable even when agents aren't coordinating (if environment dynamics are simple)

**LGDD v2:**
- Agent predicts the TEAM's next state: (z_i, g_i, a_i) → g_{t+1}
- Measures: "Can agent i predict what the TEAM will do next?"
- Better: Team behavior is only predictable when agents are acting in a coordinated, predictable manner

The cross-agent signal directly measures mutual predictability — the essence of coordination.

### A6: The Virtuous Cycle of CASVD

```
Step 0: Random policies
  → LGDD error HIGH (agents unpredictable)
  → α ≈ α_max (1.0)
  → Soft values ≈ mean(Q) (maximum exploration)
  → Agents explore freely

Step ~100k: Agents discover some coordination
  → LGDD error DROPS (agents more predictable)
  → α decreases (~0.5)
  → Soft values shift toward max(Q) (more exploitation)
  → Agents begin exploiting coordinated strategies

Step ~500k: Good coordination established
  → LGDD error LOW (agents highly predictable)
  → α ≈ α_min (0.01)
  → Soft values ≈ max(Q) (nearly greedy, like QMIX)
  → Agents execute coordinated play precisely

New team composition appears (SMACv2 procedural generation):
  → LGDD error SPIKES (new dynamics, unpredictable)
  → α jumps back up
  → Explores the new composition
  → Finds coordination → α drops → exploits
  → CL prevents forgetting of old compositions
```

This is fully automatic. No hand-tuned schedules, no phase transitions, no manual intervention.

---

## Summary

CASVD = GAT encoder + Soft Value Decomposition + Coordination-Aware Alpha + LGDD v2 + CL

Every component has a precise, justified role:
- **GAT**: entity reasoning + team communication
- **Value decomposition (QMIX mixer)**: coordination guarantee
- **Soft Q-values**: adaptive exploration via temperature α
- **LGDD v2**: coordination measurement + representation learning
- **Coordination-aware α**: automatic exploration/exploitation control
- **CL**: compositional memory across team variations

The result: an algorithm that explores when needed, coordinates when possible, and transfers across team compositions — addressing the limitations of both QMIX and SAC in a unified framework.
