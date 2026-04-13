# CASVD: Coordination-Aware Soft Value Decomposition — Complete Design Document

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [Neural Architecture: GATNSAgent](#2-neural-architecture-gatnsagent)
   - 2.1 [Observation Parsing](#21-observation-parsing)
   - 2.2 [Entity Encoding](#22-entity-encoding)
   - 2.3 [LocalEntityGAT — Agent-to-Entity Attention](#23-localentitygat--agent-to-entity-attention)
   - 2.4 [TeamGATLayer — Agent-to-Agent Attention](#24-teamgatlayer--agent-to-agent-attention)
   - 2.5 [GRU Temporal Core + Q-Head](#25-gru-temporal-core--q-head)
   - 2.6 [Full Forward Pass Summary](#26-full-forward-pass-summary)
3. [Soft Value Target Computation — The Core Innovation](#3-soft-value-target-computation--the-core-innovation)
   - 3.1 [Q-Spread Computation](#31-q-spread-computation)
   - 3.2 [Per-Agent Temperature Alpha](#32-per-agent-temperature-alpha)
   - 3.3 [Boltzmann Soft Policy (Double-Q Style)](#33-boltzmann-soft-policy-double-q-style)
   - 3.4 [Soft V-Value from Target Network](#34-soft-v-value-from-target-network)
   - 3.5 [Mixing and TD-Lambda Returns](#35-mixing-and-td-lambda-returns)
   - 3.6 [Self-Stabilizing Property — Mathematical Proof](#36-self-stabilizing-property--mathematical-proof)
   - 3.7 [Scale-Invariance Property](#37-scale-invariance-property)
4. [InfoNCE Coordination Sensor](#4-infonce-coordination-sensor)
   - 4.1 [Motivation](#41-motivation)
   - 4.2 [Architecture](#42-architecture)
   - 4.3 [InfoNCE Loss Computation](#43-infonce-loss-computation)
   - 4.4 [Coordination Signal Derivation](#44-coordination-signal-derivation)
   - 4.5 [Gradient Isolation — Critical Design Choice](#45-gradient-isolation--critical-design-choice)
5. [Per-Agent Adaptive Alpha](#5-per-agent-adaptive-alpha)
   - 5.1 [Mapping Coordination Signal to Alpha Factor](#51-mapping-coordination-signal-to-alpha-factor)
   - 5.2 [Behavioral Interpretation](#52-behavioral-interpretation)
6. [NMixer — Value Decomposition Network](#6-nmixer--value-decomposition-network)
   - 6.1 [Architecture](#61-architecture)
   - 6.2 [Monotonicity Guarantee](#62-monotonicity-guarantee)
   - 6.3 [Difference from Standard QMIX](#63-difference-from-standard-qmix)
7. [CASVD Controller](#7-casvd-controller)
   - 7.1 [Input Construction](#71-input-construction)
   - 7.2 [Action Selection](#72-action-selection)
   - 7.3 [Differences from BasicMAC](#73-differences-from-basicmac)
8. [Complete Training Flow](#8-complete-training-flow)
   - 8.1 [Episode Collection](#81-episode-collection)
   - 8.2 [Training Step — Full Sequential Walkthrough](#82-training-step--full-sequential-walkthrough)
   - 8.3 [Dual Backward Passes](#83-dual-backward-passes)
   - 8.4 [Target Network Update Strategy](#84-target-network-update-strategy)
9. [All Loss Functions](#9-all-loss-functions)
10. [Continual Learning Extension](#10-continual-learning-extension)
11. [Configuration Reference](#11-configuration-reference)
12. [Key Design Decisions and Rationale](#12-key-design-decisions-and-rationale)
13. [File Reference Map](#13-file-reference-map)

---

## 1. High-Level Overview

CASVD (Coordination-Aware Soft Value Decomposition) is a multi-agent reinforcement learning algorithm that combines four key innovations into a unified framework:

1. **Structured entity attention** (Local GAT + Team GAT) — processes heterogeneous SMACv2 observations via graph attention, allowing agents to attend to enemies, allies, and teammates in a structured way rather than treating the observation as a flat vector.

2. **Q-spread-relative soft values** — replaces the hard `max` operator in standard QMIX targets with a Boltzmann softmax whose temperature automatically adapts to the Q-value scale. This creates a self-stabilizing feedback loop that prevents value overestimation while maintaining learning signal.

3. **InfoNCE coordination sensor** — a contrastive learning module that measures how well each agent's local embedding predicts the global future team state. This provides a per-agent coordination signal without interfering with Q-value learning (gradient-isolated).

4. **Per-agent adaptive alpha** — the coordination signal drives per-agent soft value temperature: well-coordinated agents get hard-max (greedy) targets while poorly-coordinated agents get softer (more exploratory) targets.

### Conceptual Data Flow

```
SMACv2 Observation (structured entities)
         |
         v
    Entity Encoders (Linear projections per entity type)
         |
         v
    LocalEntityGAT (each agent attends to its visible entities)
         |
         v
    TeamGATLayer (agents attend to each other)
         |
         v
    GRU (temporal recurrence per agent)
         |
         v
    Q-Head (tanh-squashed Q-values per action)
         |
    +----+----+
    |         |
    v         v
  NMixer   InfoNCE Sensor (detached)
    |         |
    v         v
  Q_total   coord_signal_i
    |         |
    v         v
  TD Loss   adaptive alpha_i
              |
              v
         Soft Value Targets
```

---

## 2. Neural Architecture: GATNSAgent

**File**: `src/modules/agents/gat_ns_agent.py`

### 2.1 Observation Parsing

SMACv2 provides structured entity-based observations. Each agent's observation is a flat vector that encodes information about:
- Movement features (can move in which directions)
- Per-enemy features (health, position, shield, unit type for each visible enemy)
- Per-ally features (health, position, shield, unit type for each visible ally)
- Own features (agent's own health, shield, unit type)

The `_split_obs()` method (lines 289-305) parses this flat vector:

```
raw_obs layout:
[move_feats | enemy_1...enemy_E | ally_1...ally_A | own_feats]
     ^              ^                   ^              ^
  move_dim    E x enemy_dim       A x ally_dim     own_dim
```

```python
offset = 0
move_feats  = raw_obs[:, :, offset : offset + move_feats_dim]                    # [B, n_agents, move_dim]
offset += move_feats_dim
enemy_feats = raw_obs[:, :, offset : offset + n_enemies * enemy_feat_dim]         # [B, n_agents, E*enemy_dim]
enemy_feats = enemy_feats.reshape(B, n_agents, n_enemies, enemy_feat_dim)         # [B, n_agents, E, enemy_dim]
offset += n_enemies * enemy_feat_dim
ally_feats  = raw_obs[:, :, offset : offset + n_allies * ally_feat_dim]           # [B, n_agents, A*ally_dim]
ally_feats  = ally_feats.reshape(B, n_agents, n_allies, ally_feat_dim)            # [B, n_agents, A, ally_dim]
offset += n_allies * ally_feat_dim
own_feats   = raw_obs[:, :, offset : offset + own_feat_dim]                       # [B, n_agents, own_dim]
```

### 2.2 Entity Encoding

Each entity type is projected into a shared `hidden_dim=128` dimensional space using separate linear encoders:

```
self_node  = ReLU(W_self  * [move_feats || own_feats])     -> [B, n_agents, 128]
ally_nodes = ReLU(W_ally  * ally_feats)                    -> [B, n_agents, n_allies, 128]
enemy_nodes= ReLU(W_enemy * enemy_feats)                   -> [B, n_agents, n_enemies, 128]
```

Where:
- `W_self  in R^{(move_dim + own_dim) x 128}` — self encoder
- `W_ally  in R^{ally_dim x 128}` — ally encoder
- `W_enemy in R^{enemy_dim x 128}` — enemy encoder

An **entity mask** is constructed by checking for zero-valued entity features (dead or out-of-sight entities have all-zero features):

```
entity_mask[b, i, e] = True   if entity e is visible to agent i
                     = False  if entity e is dead/invisible (zero features)
```

All entity nodes are stacked into a single tensor:

```
entity_nodes = stack([self_node.unsqueeze(2), ally_nodes, enemy_nodes], dim=2)
             -> [B, n_agents, 1 + n_allies + n_enemies, 128]
```

### 2.3 LocalEntityGAT — Agent-to-Entity Attention

**File**: `src/modules/agents/gat_ns_agent.py`, lines 11-43

Each agent independently attends over its own observable entities using multi-head scaled dot-product attention. This is the first layer of the two-tier attention architecture.

**Parameters**:
- `n_heads = 4`
- `head_dim = hidden_dim / n_heads = 128 / 4 = 32`
- `W_Q, W_K, W_V in R^{128 x 128}` (no bias)
- `W_out in R^{128 x 128}` (with bias)

**Mathematical formulation**:

Given:
- `query_node`: the agent's self embedding `s_i in R^{128}` (shape: `[B, n_agents, 128]`)
- `entity_nodes`: all entities `E_i in R^{E x 128}` (shape: `[B, n_agents, E, 128]`)
- `entity_mask`: validity mask (shape: `[B, n_agents, E]`)

**Step 1 — Multi-head projection:**

```
Q = W_Q * s_i                    -> [B, n_agents, n_heads, head_dim]
    reshaped from [B, n_agents, 128]

K = W_K * E_i                    -> [B, n_agents, E, n_heads, head_dim]
    reshaped from [B, n_agents, E, 128]

V = W_V * E_i                    -> [B, n_agents, E, n_heads, head_dim]
    reshaped from [B, n_agents, E, 128]
```

**Step 2 — Scaled dot-product attention scores:**

```
                    Q_h * K_h^T
logits_h(i, e) = ──────────────      for each head h in {1..4}
                     sqrt(32)

logits shape: [B, n_agents, n_heads, E]
```

**Step 3 — Mask invalid entities:**

```
logits[~entity_mask] = -1e9      (effectively -infinity before softmax)
```

**Step 4 — Attention weights:**

```
                  exp(logits_h(i, e))
alpha_h(i, e) = ───────────────────────
                sum_{e'} exp(logits_h(i, e'))

alpha shape: [B, n_agents, n_heads, E]
```

**Step 5 — Weighted aggregation:**

```
context_h(i) = sum_e alpha_h(i, e) * V_h(i, e)

context shape: [B, n_agents, n_heads, head_dim]
```

**Step 6 — Concatenate heads and project:**

```
context(i) = Concat(context_1, ..., context_4)    -> [B, n_agents, 128]
output(i) = W_out * context(i)                     -> [B, n_agents, 128]
```

**Step 7 — Residual connection:**

```
local_summary(i) = output(i) + s_i                -> [B, n_agents, 128]
```

**Post-processing:**

```
local_summary = ELU(local_summary)
local_summary = LayerNorm(local_summary)           (if use_layer_norm=True)
```

If extra input features exist (e.g., agent ID one-hot, last action):
```
local_summary = local_summary + ReLU(W_extra * extras)
```

**Intuition**: Each agent builds a **local embedding** that summarizes what it observes — nearby enemies, allies, and its own state — using attention to weight the importance of each entity. A nearby low-health enemy might get high attention weight; a distant full-health ally might get low weight.

### 2.4 TeamGATLayer — Agent-to-Agent Attention

**File**: `src/modules/agents/gat_ns_agent.py`, lines 46-71

After each agent has its local summary, all agents attend to each other to share information. This uses the classical GAT (Graph Attention Network) formulation with additive attention.

**Parameters**:
- `n_heads = 4`
- `head_dim = hidden_dim / n_heads = 128 / 4 = 32`
- `W in R^{128 x 128}` (no bias) — shared linear projection
- `a_src in R^{n_heads x head_dim}` — source attention vector per head
- `a_dst in R^{n_heads x head_dim}` — destination attention vector per head
- Activation: `LeakyReLU(negative_slope=0.2)`

**Mathematical formulation**:

Given `local_summary in R^{n_agents x 128}`:

**Step 1 — Project to multi-head space:**

```
h_i = W * local_summary_i        -> [B, n_agents, n_heads, head_dim]
```

**Step 2 — Additive attention coefficients (GAT-style):**

For each head `k`:

```
e_src_i^k = h_i^k . a_src^k     -> scalar per (agent, head)
e_dst_j^k = h_j^k . a_dst^k     -> scalar per (agent, head)

e_{ij}^k = LeakyReLU(e_src_i^k + e_dst_j^k)

Shape of e: [B, n_agents, n_agents, n_heads]
```

This is the standard GAT attention mechanism where the attention between agent i and agent j is decomposed into additive source and destination terms.

**Step 3 — Attention weights (softmax over source agents j):**

```
                    exp(e_{ij}^k)
alpha_{ij}^k = ─────────────────────
                sum_{j'} exp(e_{ij'}^k)

Shape: [B, n_agents, n_agents, n_heads]
```

**Step 4 — Weighted aggregation per head:**

```
team_summary_i^k = sum_j alpha_{ij}^k * h_j^k

Shape: [B, n_heads, n_agents, head_dim]
```

Implementation uses matrix multiplication:
```
attn_w:  [B, n_heads, n_agents, n_agents]
h_perm:  [B, n_heads, n_agents, head_dim]
out = matmul(attn_w, h_perm)  ->  [B, n_heads, n_agents, head_dim]
```

**Step 5 — Concatenate heads:**

```
team_summary_i = Concat(team_summary_i^1, ..., team_summary_i^4)
               -> [B, n_agents, 128]
```

**Post-processing:**

```
team_summary = ELU(team_summary)
team_summary = LayerNorm(team_summary)       (if use_layer_norm=True)
```

**Intuition**: The team GAT allows agents to form a **team-aware embedding**. Agent i can learn what agents j and k are planning by attending to their local summaries. The additive attention mechanism learns which agent pairs need to communicate more.

### 2.5 GRU Temporal Core + Q-Head

After the two-tier attention, each agent's local and team summaries are concatenated and fed through a shared GRU for temporal reasoning:

**GRU Input Construction:**

```
gru_input_i = [local_summary_i || team_summary_i]     -> [B, n_agents, 256]
```

Flattened for batch processing:
```
flat_input = gru_input.reshape(B * n_agents, 256)
flat_hidden = hidden_states.reshape(B * n_agents, 128)
```

**GRU Forward:**

```
z_t = sigmoid(W_z * [flat_input || h_{t-1}])          (update gate)
r_t = sigmoid(W_r * [flat_input || h_{t-1}])          (reset gate)
h_candidate = tanh(W_h * [flat_input || (r_t . h_{t-1})])
h_t = (1 - z_t) . h_{t-1} + z_t . h_candidate

h_t shape: [B * n_agents, 128]
```

**Q-Head with Tanh Squashing:**

```
Q_raw = W_q * h_t + b_q            -> [B * n_agents, n_actions]
Q = tanh(Q_raw) * q_output_scale   -> Q in [-2.0, +2.0]
```

Reshaped back:
```
Q -> [B, n_agents, n_actions]
h_t -> [B, n_agents, 128]
```

**Weight initialization**:
- Encoder weights: orthogonal with `gain=1.0`
- Q-head weights: orthogonal with `gain=0.1` (small initial Q-values, near-uniform initial soft policies)

### 2.6 Full Forward Pass Summary

```
Input: obs [B, n_agents, obs_dim], hidden [B, n_agents, 128]
                    |
                    v
            _split_obs(obs)
                    |
        +-----------+-----------+---------------+
        |           |           |               |
   move_feats  enemy_feats  ally_feats     own_feats
        |           |           |               |
        v           v           v               v
   W_self(move||own) W_enemy    W_ally
        |           |           |
        v           v           v
   self_node    enemy_nodes  ally_nodes         (all in R^128)
        |           |           |
        +-----+-----+-----------+
              |
              v
        entity_nodes [B, n_agents, E, 128]
              |
              v
    +-------------------+
    | LocalEntityGAT    |
    | (scaled dot-prod  |
    |  multi-head attn) |
    +-------------------+
              |
              v
    local_summary [B, n_agents, 128]   ---> saved for InfoNCE (detached)
              |
              v
    +-------------------+
    | TeamGATLayer      |
    | (additive GAT     |
    |  multi-head attn) |
    +-------------------+
              |
              v
    team_summary [B, n_agents, 128]
              |
              v
    [local_summary || team_summary]  -> [B, n_agents, 256]
              |
              v
    +-------------------+
    | GRU               |
    | (shared across    |
    |  all agents)      |
    +-------------------+
              |
              v
    h_t [B, n_agents, 128]
              |
              v
    +-------------------+
    | Q-Head            |
    | tanh * 2.0        |
    +-------------------+
              |
              v
    Q [B, n_agents, n_actions]     Q in [-2.0, +2.0]
```

---

## 3. Soft Value Target Computation — The Core Innovation

**File**: `src/learners/casvd_learner.py`, lines 182-242

In standard QMIX, targets are computed using the hard maximum:

```
V_target(s, i) = max_a Q_target(s, i, a)
```

CASVD replaces this with a **Boltzmann soft maximum** whose temperature is proportional to the Q-value spread, creating a self-stabilizing mechanism.

### 3.1 Q-Spread Computation

For each agent `i` at state `s` and timestep `t`, compute the spread of Q-values across available actions:

```
Q_online(s, i, a)  for all actions a in A_i(s)    (from online network, detached)
```

Mask unavailable actions:
```
Q_masked(s, i, a) = Q_online(s, i, a)  if avail(s, i, a) = 1
                  = -1e10              if avail(s, i, a) = 0
```

Compute spread:
```
Q_max(s, i) = max_a Q_masked(s, i, a)

                    sum_{a: avail=1} Q_online(s, i, a)
Q_mean(s, i) = ─────────────────────────────────────────
                    |{a : avail(s, i, a) = 1}|

DeltaQ(s, i) = max(Q_max(s, i) - Q_mean(s, i), 1e-6)
```

The `1e-6` floor prevents division by zero when all Q-values are equal.

**Shape**: `DeltaQ -> [B, T, n_agents, 1]`

### 3.2 Per-Agent Temperature Alpha

The effective temperature for agent `i` is:

```
alpha_i(s, t) = max( alpha_factor_i * DeltaQ(s, i, t) , alpha_floor )
```

Where:
- `alpha_factor_i` is the per-agent adaptive factor driven by coordination signal (see Section 5)
- `alpha_floor = 0.005` is a minimum temperature to prevent numerical issues
- `DeltaQ(s, i, t)` is the Q-spread from Section 3.1

**Shape**: `alpha_per -> [B, T, n_agents, 1]`

### 3.3 Boltzmann Soft Policy (Double-Q Style)

The soft policy is computed using the **online** network's Q-values (Double-Q: online selects, target evaluates):

```
                     exp(Q_online(s, i, a) / alpha_i)
pi_soft(a | s, i) = ──────────────────────────────────────
                     sum_{a'} exp(Q_online(s, i, a') / alpha_i)
```

Unavailable actions are masked out before softmax (set to `-1e10`).

**Shape**: `pi_soft -> [B, T, n_agents, n_actions]`

### 3.4 Soft V-Value from Target Network

The soft value for each agent is the expectation of target Q-values under the soft policy:

```
V_soft(s, i) = sum_a pi_soft(a | s, i) * Q_target(s, i, a)
             = E_{a ~ pi_soft} [ Q_target(s, i, a) ]
```

**Shape**: `V_soft -> [B, T, n_agents]`

This is the **Double-Q adaptation for soft values**:
- The **online** network determines the soft policy (which actions to weight)
- The **target** network provides the Q-value estimates (what those actions are worth)
- This decoupling prevents the overestimation bias that would occur if the same network both selected and evaluated

### 3.5 Mixing and TD-Lambda Returns

The per-agent soft values are mixed into a joint team value:

```
Q_total_target = NMixer(V_soft, state)      -> [B, T, 1]
```

Then TD(lambda) returns are computed:

```
G_T = Q_total_target_T                      (bootstrap from final step)

For t = T-1, T-2, ..., 0:
    G_t = r_t + gamma * (1 - d_t) * [ lambda * G_{t+1} + (1 - lambda) * Q_total_target_{t+1} ]

Where:
    gamma = 0.99     (discount factor)
    lambda = 0.6     (TD-lambda mixing parameter)
    d_t = terminated  (episode done flag)
```

**Shape**: `targets -> [B, T-1, 1]`

### 3.6 Self-Stabilizing Property — Mathematical Proof

The Q-spread-relative temperature creates a **negative feedback loop** that prevents both value explosion and value collapse:

**Feedback Loop:**

```
Step 1: Soft values suppress extreme Q-values
        -> Q-value range shrinks over time

Step 2: Smaller Q range -> smaller DeltaQ = max(Q) - mean(Q)

Step 3: Smaller DeltaQ -> smaller alpha = factor * DeltaQ

Step 4: Smaller alpha -> pi_soft approaches hard argmax
        -> less suppression of extreme values
        -> learning signal is preserved

Step 5: If Q-values grow again -> DeltaQ grows -> alpha grows
        -> more suppression kicks in
```

This is a **self-correcting mechanism**:
- If soft values over-suppress: alpha shrinks, softmax sharpens, returns toward hard-max behavior
- If Q-values explode: alpha grows, softmax softens, dampening the explosion

**Equilibrium condition**: The system finds a natural equilibrium where the softmax temperature is just right for the current Q-value scale.

### 3.7 Scale-Invariance Property

The soft value formulation is **invariant to uniform scaling of Q-values**:

**Proof:**

Suppose all Q-values scale by constant `c > 0`:
```
Q'(s, i, a) = c * Q(s, i, a)   for all s, i, a
```

Then:
```
DeltaQ' = max(Q') - mean(Q') = c * max(Q) - c * mean(Q) = c * DeltaQ
alpha'  = factor * DeltaQ' = factor * c * DeltaQ = c * alpha
```

The softmax argument becomes:
```
Q'(s, i, a) / alpha' = c * Q(s, i, a) / (c * alpha) = Q(s, i, a) / alpha
```

Therefore `pi_soft` is **unchanged** regardless of Q-value scale. This means the algorithm is robust across:
- Different training stages (Q-values naturally grow during learning)
- Different reward scales across maps
- Different numbers of agents (which affects reward magnitude)

---

## 4. InfoNCE Coordination Sensor

**File**: `src/modules/predictors/infonce_predictor.py`

### 4.1 Motivation

In multi-agent systems, some agents may be well-coordinated with the team (their local information is predictive of the team's future) while others may be acting independently or confused. CASVD uses InfoNCE contrastive learning to **measure** this coordination without **interfering** with Q-value learning.

The key question InfoNCE answers: **"How well does agent i's local embedding predict what the global team state will be at the next timestep?"**

- If well: agent i is coordinated (its local decisions align with team outcomes)
- If poorly: agent i is uncoordinated (its local view doesn't capture team dynamics)

### 4.2 Architecture

The InfoNCE predictor is minimal — a single linear projection:

```
W: Linear(hidden_dim -> hidden_dim, no bias)     R^{128 x 128}
   Initialized: orthogonal
   Purpose: learn which features of local embeddings predict global outcomes
```

Temperature parameter: `tau = 0.1` (controls sharpness of contrastive distribution)

### 4.3 InfoNCE Loss Computation

**File**: `src/modules/predictors/infonce_predictor.py`, lines 34-80

**Inputs (all detached from encoder — no gradient flows back to GAT):**

```
h_i:    [B, n_agents, D]     agent local embeddings at time t (from LocalEntityGAT)
g_pos:  [B, D]               real global team state at time t+1
g_neg:  [B, K, D]            K=15 fake global states from random timesteps
```

Where the global team state is computed as:
```
g(t) = mean_i(local_summary_i(t))     (mean pool over all agents)
```

**Step 1 — Project local embeddings:**

```
h_proj = W * h_i                       -> [B, n_agents, D]
```

**Step 2 — L2 normalize all vectors to unit sphere:**

```
h_proj_n = h_proj / ||h_proj||_2       -> unit vectors in R^D
g_pos_n  = g_pos  / ||g_pos||_2
g_neg_n  = g_neg  / ||g_neg||_2
```

L2 normalization ensures similarity scores are bounded in `[-1/tau, +1/tau]` regardless of embedding magnitude, preventing the loss from being dominated by vector norms.

**Step 3 — Compute cosine similarity scores (scaled by temperature):**

```
                 h_proj_n(b, i) . g_pos_n(b)
score_pos(b,i) = ─────────────────────────────     -> [B, n_agents]
                           tau

                 h_proj_n(b, i) . g_neg_n(b, k)
score_neg(b,i,k) = ─────────────────────────────   -> [B, n_agents, K]
                           tau
```

Where `.` denotes dot product and `tau = 0.1`.

**Step 4 — Concatenate into logits:**

```
logits = [score_pos.unsqueeze(-1), score_neg]      -> [B, n_agents, K+1]

logits[:, :, 0]   = positive score
logits[:, :, 1:]  = K negative scores
```

**Step 5 — Cross-entropy loss (positive label is index 0):**

```
labels = zeros(B, n_agents)    (class 0 = positive)

L_InfoNCE(b, i) = -log( exp(score_pos(b,i)) / sum_{k=0}^{K} exp(logits(b,i,k)) )
                = CrossEntropy(logits[b,i,:], label=0)
```

**Shape**: `loss -> [B, n_agents]`

**Range**:
```
Minimum: 0          (perfect prediction — always picks positive)
Maximum: log(K+1)   = log(16) ~ 2.77   (random chance)
```

### 4.4 Coordination Signal Derivation

**File**: `src/learners/casvd_learner.py`, lines 337-344

The per-timestep InfoNCE losses are aggregated across the episode:

**Step 1 — Collect per-timestep losses:**

For each timestep `t in {0, ..., T-2}`:
```
h_i     = local_summary[t].detach()                 -> [B, n_agents, D]
g_pos   = mean_agents(local_summary[t+1]).detach()   -> [B, D]
g_neg_k = mean_agents(local_summary[neg_idx_k]).detach()  for k in {1..K}
                                                      -> [B, K, D]

loss_t = InfoNCEPredictor(h_i, g_pos, g_neg)         -> [B, n_agents]
```

Negative indices are sampled uniformly from `{0, ..., T-1} \ {t+1}` (any timestep except the true next one).

**Step 2 — Average across time (masked for variable-length episodes):**

```
all_losses = stack(loss_0, ..., loss_{T-2})           -> [B, T-1, n_agents]
mask = episode_mask                                    -> [B, T-1, 1]

                  sum_t (loss_t * mask_t)
per_agent_loss = ─────────────────────────            -> [B, n_agents]
                      sum_t mask_t
```

**Step 3 — Compute coordination signal:**

```
max_possible_loss = log(K + 1) = log(16)

coord_signal_batch = clamp( mean_B(per_agent_loss) / log(K+1) , 0, 1)
                   -> [n_agents]
```

**Step 4 — EMA smoothing:**

```
coord_signals = tau * coord_signals_old + (1 - tau) * coord_signal_batch

Where tau = 0.999 (very strong smoothing for stability)
```

**Interpretation:**
```
coord_signal_i ~ 0.0  =>  Agent i perfectly predicts global future  =>  well-coordinated
coord_signal_i ~ 1.0  =>  Agent i cannot predict global future      =>  poorly-coordinated
```

### 4.5 Gradient Isolation — Critical Design Choice

**This is one of the most important design decisions in CASVD.**

The InfoNCE loss trains **only** the predictor weight matrix `W`. The encoder embeddings (`local_summary`) are `.detach()`-ed before being passed to the predictor:

```python
all_local = stack([lat["local_summary"].detach() for lat in all_latents])
```

Two completely separate backward passes enforce this:

```
Pass 1: lgdd_optimizer.zero_grad()
        infonce_loss.backward()      -> updates only W
        lgdd_optimizer.step()

Pass 2: main_optimizer.zero_grad()
        td_loss.backward()           -> updates encoder + GRU + Q-head + mixer
        main_optimizer.step()
```

**Why this matters**: If InfoNCE gradients flowed into the encoder, they would compete with Q-value learning gradients. The encoder would be torn between:
- Making Q-values accurate (TD loss wants)
- Making embeddings predictive of global future (InfoNCE wants)

These objectives can conflict. By detaching, the InfoNCE predictor acts as a **read-only sensor** — it observes the encoder's embeddings to measure coordination, but never pushes the encoder in any direction. The encoder is free to optimize purely for Q-value accuracy.

---

## 5. Per-Agent Adaptive Alpha

**File**: `src/learners/casvd_learner.py`, lines 206-214

### 5.1 Mapping Coordination Signal to Alpha Factor

The coordination signal from InfoNCE is mapped linearly to a per-agent alpha factor:

```
alpha_factor_i = alpha_min + (alpha_max - alpha_min) * coord_signal_i
               = 0.0 + (1.0 - 0.0) * coord_signal_i
               = coord_signal_i
```

This alpha factor then scales the Q-spread to produce the effective temperature:

```
alpha_per_i(s, t) = alpha_factor_i * DeltaQ_i(s, t)
alpha_eff_i(s, t) = max(alpha_per_i(s, t), alpha_floor)

Where alpha_floor = 0.005
```

The complete chain from InfoNCE to soft value temperature:

```
InfoNCE loss per agent
        |
        v
coord_signal_i = clamp(loss_i / log(K+1), 0, 1)    EMA smoothed
        |
        v
alpha_factor_i = 0.0 + 1.0 * coord_signal_i        linear mapping
        |
        v
alpha_per_i = alpha_factor_i * DeltaQ_i(s,t)        scale by Q-spread
        |
        v
alpha_eff_i = max(alpha_per_i, 0.005)               floor for stability
        |
        v
pi_soft(a|s,i) = softmax(Q_online(s,i,:) / alpha_eff_i)
```

### 5.2 Behavioral Interpretation

| Agent State | coord_signal | alpha_factor | DeltaQ (example) | alpha_eff | Softmax behavior |
|-------------|-------------|-------------|------------------|-----------|-----------------|
| Well-coordinated | ~0.0 | ~0.0 | 0.04 | 0.005 (floor) | Near hard-max (exploit) |
| Moderately coordinated | ~0.3 | ~0.3 | 0.04 | 0.012 | Slightly soft |
| Poorly coordinated | ~0.8 | ~0.8 | 0.04 | 0.032 | Quite soft (explore) |
| Completely uncoordinated | ~1.0 | ~1.0 | 0.04 | 0.040 | Very soft (maximum exploration) |

**Intuition**:
- **Well-coordinated agents** (low InfoNCE loss) get near-zero alpha, meaning their soft targets approach the hard maximum. These agents already "understand" the team dynamics, so they should exploit their knowledge greedily.
- **Poorly-coordinated agents** (high InfoNCE loss) get higher alpha, meaning their soft targets are more diffuse. These agents are confused about team dynamics, so giving them softer targets provides the mixer with more diverse Q-values to learn from, and prevents premature convergence to a suboptimal joint policy.

---

## 6. NMixer — Value Decomposition Network

**File**: `src/modules/mixers/nmix.py`

### 6.1 Architecture

The NMixer is a state-conditioned two-layer hypernetwork that combines per-agent Q-values into a joint Q-total while maintaining monotonicity:

**Hypernetworks (generate weights from state):**

```
hyper_w1: state -> w1 in R^{n_agents x embed_dim}     (via Linear or 2-layer MLP)
hyper_b1: state -> b1 in R^{embed_dim}                 (via Linear)
hyper_w2: state -> w2 in R^{embed_dim x 1}             (via Linear or 2-layer MLP)
hyper_b2: state -> b2 in R^{1}                          (via 2-layer MLP with ReLU)
```

Where `embed_dim = 32` and `hypernet_embed = 64` (hidden dim of 2-layer hypernets).

### 6.2 Forward Pass — Mathematical Formulation

**Input:**
```
Q_agents: [B, T, n_agents]     per-agent Q-values (or V-soft values)
state:    [B, T, state_dim]    global state
```

**Reshape for batch processing:**
```
Q_agents -> [B*T, 1, n_agents]
state    -> [B*T, state_dim]
```

**Layer 1:**
```
w1 = pos_func(hyper_w1(state))    -> [B*T, n_agents, embed_dim]
b1 = hyper_b1(state)              -> [B*T, 1, embed_dim]

hidden = ELU(Q_agents @ w1 + b1)  -> [B*T, 1, embed_dim]
```

**Layer 2:**
```
w2 = pos_func(hyper_w2(state))    -> [B*T, embed_dim, 1]
b2 = hyper_b2(state)              -> [B*T, 1, 1]

Q_total = hidden @ w2 + b2        -> [B*T, 1, 1]
```

**Output:**
```
Q_total -> [B, T, 1]
```

**Full mathematical form:**
```
Q_total(s) = b2(s) + ELU(Q_agents * w1(s) + b1(s)) * w2(s)

Where:
    w1(s) = |f_w1(s)|      in R^{n_agents x 32}   (non-negative)
    w2(s) = |f_w2(s)|      in R^{32 x 1}           (non-negative)
    b1(s) = f_b1(s)        in R^{32}                (unconstrained)
    b2(s) = f_b2(s)        in R^{1}                 (unconstrained)
```

### 6.2 Monotonicity Guarantee

The positivity function (`pos_func`) applied to `w1` and `w2` ensures:

```
w1 >= 0  and  w2 >= 0

=> dQ_total / dQ_i >= 0   for all agents i
```

This means: **if any individual agent improves its Q-value, the total team value cannot decrease**. This is the Individual-Global-Max (IGM) principle that enables decentralized execution — each agent can greedily maximize its own Q-value during execution, and the team value is guaranteed to improve.

The `pos_func` options are:
- `abs`: `w = |w|` (default, used in CASVD)
- `softplus`: `w = log(1 + exp(w))`
- `quadratic`: `w = w^2`

### 6.3 Difference from Standard QMIX

| Feature | Standard QMIX | NMixer |
|---------|--------------|--------|
| Bias term | `V(s)` separate network | State-dependent `b2(s)` via hypernetwork |
| Layer 2 bias | State value function `V(s) = MLP(state)` | `hyper_b2(state)` via same hypernetwork pattern |
| Weight constraint | `abs()` | Configurable: `abs`, `softplus`, `quadratic` |
| Architecture | 2-layer hypernet for weights | Same pattern, slightly different parameterization |

The key structural difference is that NMixer uses `hyper_b2(state)` instead of a separate value function `V(s)`, making the architecture more uniform.

---

## 7. CASVD Controller

**File**: `src/controllers/casvd_controller.py`

### 7.1 Input Construction

The CASVD controller builds agent inputs by concatenating:

```python
inputs = []

# 1. Current observation
inputs.append(obs)                                          # [B, n_agents, obs_dim]

# 2. Last action (one-hot) — provides action history
if obs_last_action:
    if t == 0:
        inputs.append(zeros(B, n_agents, n_actions))        # zeros at episode start
    else:
        inputs.append(actions_onehot[:, t-1])               # [B, n_agents, n_actions]

# 3. Agent ID (one-hot) — enables parameter sharing with agent-specific behavior
if obs_agent_id:
    inputs.append(eye(n_agents).expand(B, -1, -1))          # [B, n_agents, n_agents]

final_input = cat(inputs, dim=-1)
# Shape: [B, n_agents, obs_dim + n_actions + n_agents]
```

### 7.2 Action Selection

```python
# During data collection (epsilon-greedy):
q_values = forward(ep_batch, t)               # [B, n_agents, n_actions]
# Masking of unavailable actions happens inside action_selector
actions = action_selector.select_action(q_values, avail_actions, t_env, test_mode)

# During test (greedy):
actions = argmax(q_values * avail_mask)       # pure exploitation
```

Epsilon schedule:
```
epsilon(t) = max(epsilon_finish, epsilon_start - t / epsilon_anneal_time)
           = max(0.05, 1.0 - t / 100000)
```

### 7.3 Differences from BasicMAC

1. **Raw Q-value output**: CASVD controller returns raw, unmasked Q-values. Masking for unavailable actions is handled separately by the learner and action selector. This enables clean softmax computation in the soft value formulation.

2. **Latent access**: The `forward_with_latents()` method also returns encoder latents (local_summary, team_summary) needed for the InfoNCE coordination sensor:
   ```python
   q_values, latents = mac.forward_with_latents(batch, t)
   # latents = {"local_summary": [B, n_agents, 128],
   #            "team_summary":  [B, n_agents, 128]}
   ```

3. **No internal masking/softmax**: Unlike BasicMAC which applies softmax for `pi_logits` output type, CASVD controller outputs raw Q-values (`agent_output_type: "q"`) and lets the learner handle all transformations.

---

## 8. Complete Training Flow

### 8.1 Episode Collection

```
ParallelRunner spawns 8 parallel StarCraft II environments.

For each environment simultaneously:
    1. env.reset()
       -> SMACv2 samples team composition from config
          (e.g., 3 stalkers + 2 zealots, random positions)

    2. For t = 0, 1, ..., episode_limit:
        a. state = env.get_state()                    # global state
        b. obs = env.get_obs()                        # per-agent observations
        c. avail = env.get_avail_actions()            # per-agent action masks

        d. MAC.forward(batch, t)                      # GATNSAgent forward
           -> Q_values [1, n_agents, n_actions]

        e. actions = epsilon_greedy(Q_values, avail)  # action selection

        f. reward, terminated, info = env.step(actions)

        g. Store transition: (state, obs, actions, avail, reward, terminated)

    3. Return episode batch [8 episodes]

    4. Insert into ReplayBuffer (capacity=5000 episodes)
```

### 8.2 Training Step — Full Sequential Walkthrough

**File**: `src/learners/casvd_learner.py`, lines 128-436

```
========================================================================
STEP 1: BATCH EXTRACTION (lines 129-143)
========================================================================

Sample batch_size=128 episodes from replay buffer.
Extract tensors:

    rewards:       [128, T-1, 1]
    actions:       [128, T-1, n_agents, 1]     (integer action indices)
    terminated:    [128, T-1, 1]
    mask:          [128, T-1, 1]               (1 where data is valid)
    avail_actions: [128, T, n_agents, n_actions]

========================================================================
STEP 2: ONLINE NETWORK FORWARD PASS (lines 148-160)
========================================================================

Initialize hidden states to zeros.

For t = 0 to T:
    inputs = build_inputs(batch, t)            # obs + last_action + agent_id
    q_t, latents_t = agent.forward_with_latents(inputs, hidden)

    mac_out.append(q_t)                        # [128, n_agents, n_actions]
    all_latents.append(latents_t)              # {local_summary, team_summary}

mac_out = stack(mac_out, dim=1)                # [128, T, n_agents, n_actions]

========================================================================
STEP 3: TARGET NETWORK FORWARD PASS (lines 162-176)
========================================================================

Same procedure but with target_mac (frozen weights):

For t = 0 to T:
    target_q_t = target_agent.forward(inputs, target_hidden)
    target_mac_out.append(target_q_t)

target_mac_out = stack(...)                    # [128, T, n_agents, n_actions]

========================================================================
STEP 4: SOFT VALUE TARGET COMPUTATION (lines 182-242)
========================================================================

--- 4a: Q-spread (lines 193-202) ---

online_q = mac_out.detach()
online_q[avail_actions == 0] = -1e10

q_max = max(online_q, dim=-1, keepdim=True)     # [128, T, n_agents, 1]
q_count = sum(avail_actions, dim=-1, keepdim=True)
q_mean = sum(online_q * avail_actions, dim=-1) / q_count   # [128, T, n_agents, 1]
q_spread = max(q_max - q_mean, 1e-6)            # [128, T, n_agents, 1]

--- 4b: Per-agent alpha (lines 206-214) ---

If use_adaptive_alpha and coord_signals exist:
    af = coord_signals.view(1, 1, n_agents, 1)  # broadcast [1, 1, N, 1]
    af = alpha_min + (alpha_max - alpha_min) * af
    alpha_per = af * q_spread                    # [128, T, n_agents, 1]
Else:
    alpha_per = alpha_factor * q_spread

alpha_per = max(alpha_per, alpha_floor=0.005)

--- 4c: Soft policy from online network (lines 221-222) ---

online_for_soft = mac_out.detach()
online_for_soft[avail_actions == 0] = -1e10
pi_soft = softmax(online_for_soft / alpha_per, dim=-1)   # [128, T, n_agents, n_actions]

--- 4d: Soft V-value from target network (lines 228-230) ---

target_for_soft = target_mac_out
target_for_soft[avail_actions == 0] = 0
v_soft = sum(pi_soft * target_for_soft, dim=-1)  # [128, T, n_agents]

--- 4e: Mix through NMixer (lines 233-235) ---

Q_total_target = NMixer(v_soft, state)            # [128, T, 1]

--- 4f: TD-lambda returns (lines 237-242) ---

targets = build_td_lambda_targets(
    rewards, terminated, mask,
    Q_total_target, n_agents,
    gamma=0.99, td_lambda=0.6
)
# targets: [128, T-1, 1]

========================================================================
STEP 5: MIX CHOSEN Q-VALUES (lines 267-270)
========================================================================

chosen_q = gather(mac_out[:, :-1], dim=-1, index=actions)
         -> [128, T-1, n_agents]         (Q-value of action actually taken)

Q_total_online = NMixer(chosen_q, state[:, :-1])
               -> [128, T-1, 1]

========================================================================
STEP 6: TD LOSS (lines 275-277)
========================================================================

td_error = Q_total_online - targets.detach()     # [128, T-1, 1]
masked_td_error = 0.5 * (td_error ** 2) * mask   # masked MSE
td_loss = sum(masked_td_error) / sum(mask)        # scalar

========================================================================
STEP 7: InfoNCE COORDINATION SENSOR (lines 290-351)
========================================================================

--- 7a: Collect embeddings (DETACHED) (lines 295-297) ---

all_local = stack([lat["local_summary"].detach() for lat in all_latents])
          -> [T, 128, n_agents, D]       D=128

--- 7b: Compute global team state (line 300) ---

g_all = mean(all_local, dim=2)           -> [T, 128, D]

--- 7c: Per-timestep InfoNCE (lines 306-322) ---

For t = 0 to T-2:
    h_i   = all_local[t]                 -> [128, n_agents, D]
    g_pos = g_all[t+1]                   -> [128, D]

    neg_indices = sample K=15 from {0..T-1} \ {t+1}
    g_neg = stack([g_all[idx] for idx in neg_indices])
          -> [128, K, D]

    loss_t = InfoNCEPredictor(h_i, g_pos, g_neg)
           -> [128, n_agents]

    all_per_agent_loss.append(loss_t)

--- 7d: Average across time (lines 325-330) ---

all_losses = stack(all_per_agent_loss)    -> [128, T-1, n_agents]
mask_exp = mask.unsqueeze(-1)             -> [128, T-1, 1]
per_agent_loss = sum(all_losses * mask, dim=1) / sum(mask, dim=1)
               -> [128, n_agents]

--- 7e: Overall InfoNCE loss (lines 333-334) ---

infonce_loss = mean(per_agent_loss)       -> scalar

--- 7f: Update coordination signals (lines 337-344) ---

coord_batch = (mean_B(per_agent_loss) / log(K+1)).clamp(0, 1)
            -> [n_agents]

coord_signals = 0.999 * coord_signals + 0.001 * coord_batch

--- 7g: Update scalar alpha_factor (lines 348-351) ---

alpha_factor = alpha_min + (alpha_max - alpha_min) * mean(coord_signals)
```

### 8.3 Dual Backward Passes

**File**: `src/learners/casvd_learner.py`, lines 368-378

The two losses are backpropagated through completely separate computational graphs:

```
========================================================================
BACKWARD PASS 1: InfoNCE PREDICTOR (lines 368-372)
========================================================================

lgdd_optimizer.zero_grad()
infonce_loss.backward()

    Gradient flows through:
        InfoNCEPredictor.W  <-- only parameter updated
        (all encoder embeddings were .detach()-ed)

clip_grad_norm_(dynamics_params, max_norm=10)
lgdd_optimizer.step()          # lr=0.0003, Adam

========================================================================
BACKWARD PASS 2: MAIN NETWORK (lines 375-378)
========================================================================

main_optimizer.zero_grad()
td_loss.backward()

    Gradient flows through:
        Q-Head (W_q, b_q)
        GRU (W_z, W_r, W_h)
        TeamGATLayer (W, a_src, a_dst)
        LocalEntityGAT (W_Q, W_K, W_V, W_out)
        Entity encoders (W_self, W_ally, W_enemy)
        NMixer hypernetworks (hyper_w1, hyper_b1, hyper_w2, hyper_b2)

grad_norm = clip_grad_norm_(main_params, max_norm=10)
main_optimizer.step()          # lr=0.001, Adam
```

### 8.4 Target Network Update Strategy

**File**: `src/learners/casvd_learner.py`, lines 381-393

**Hard update (default, every 200 episodes):**
```
if (episode_num - last_target_update) >= target_update_interval:
    target_mac.load_state_dict(mac.state_dict())
    target_mixer.load_state_dict(mixer.state_dict())
    last_target_update = episode_num
```

**Soft update (alternative, if tau < 1):**
```
For each (target_param, online_param):
    target_param = (1 - tau) * target_param + tau * online_param
```

**Teacher MAC EMA update (if continual learning enabled):**
```
For each (teacher_param, student_param):
    teacher_param = (1 - cl_teacher_ema_tau) * teacher_param + cl_teacher_ema_tau * student_param
    Where cl_teacher_ema_tau = 0.002 (fast adaptation)
```

---

## 9. All Loss Functions

### Primary: TD Loss

```
L_TD = (1/N) * sum_{b,t} [ 0.5 * (Q_total(s_t) - G_t)^2 * mask(b,t) ]

Where:
    Q_total(s_t) = NMixer(Q_chosen(s_t), state_t)
    G_t = TD-lambda target with gamma=0.99, lambda=0.6
    N = sum(mask)  (number of valid timesteps)
```

**Trains**: Entire encoder (GAT layers) + GRU + Q-head + NMixer
**Optimizer**: `main_optimizer` (Adam, lr=0.001, eps=1e-7)

### Secondary: InfoNCE Loss

```
L_InfoNCE = (1/(B*N)) * sum_{b,i} [ -log( exp(s_pos / tau) / (exp(s_pos/tau) + sum_k exp(s_neg_k/tau)) ) ]

Where:
    s_pos = normalize(W * h_i) . normalize(g_{t+1}) / tau
    s_neg_k = normalize(W * h_i) . normalize(g_{neg_k}) / tau
    tau = 0.1, K = 15
```

**Trains**: Only `InfoNCEPredictor.W` (single linear layer)
**Optimizer**: `lgdd_optimizer` (Adam, lr=0.0003, eps=1e-7)

### Optional: Continual Learning Distillation Loss

```
L_CL = KL(pi_teacher || pi_student)
     = sum_a pi_teacher(a) * (log pi_teacher(a) - log pi_student(a))

Where:
    pi_teacher = softmax(Q_teacher / cl_temperature)     cl_temperature=0.1
    pi_student = softmax(Q_student / cl_temperature)
    Q_teacher from EMA teacher network (detached)
    Applied only on memory portion of replay buffer
```

**Weight**: `cl_distill_weight = 0.05`
**Trains**: Encoder + Q-head (added to td_loss before backward)
**Optimizer**: `main_optimizer`

### Total Loss

```
L_total_main = L_TD + cl_distill_weight * L_CL    (if CL enabled)
L_total_lgdd = L_InfoNCE                           (separate backward pass)
```

---

## 10. Continual Learning Extension

**File**: `src/learners/casvd_learner.py`, lines 441-485

When `cl_enabled=True`, CASVD supports continual learning scenarios where the task distribution changes over time (e.g., different map compositions).

### Teacher Network

An EMA (Exponential Moving Average) copy of the main network serves as the teacher:

```
teacher_mac = deepcopy(mac)       (initialized at start)

After each training step:
    teacher_param = (1 - 0.002) * teacher_param + 0.002 * student_param
```

The teacher adapts slowly, providing a stable reference distribution.

### Distillation on Memory Buffer

The replay buffer is split into:
- **Current batch** (80%): recent experiences, trained normally with TD loss
- **Memory batch** (20%): older experiences, additionally trained with distillation

For memory samples:

```
Q_student = online_mac(obs_memory)              -> [B_mem, n_agents, n_actions]
Q_teacher = teacher_mac(obs_memory).detach()    -> [B_mem, n_agents, n_actions]

pi_student = softmax(Q_student / 0.1)
pi_teacher = softmax(Q_teacher / 0.1)

L_distill = mean( sum_a pi_teacher * (log pi_teacher - log pi_student) )
          = mean( KL(pi_teacher || pi_student) )
```

This prevents catastrophic forgetting by ensuring the student doesn't drift too far from previously learned behaviors.

### Reservoir Buffer

A separate `ReservoirReplayBuffer` (reservoir sampling) maintains a uniform sample from all past episodes, ensuring old experiences aren't completely overwritten.

---

## 11. Configuration Reference

**File**: `src/config/algs/casvd.yaml`

### Action Selection
```yaml
action_selector: "epsilon_greedy"
epsilon_start: 1.0                    # initial exploration rate
epsilon_finish: 0.05                  # final exploration rate
epsilon_anneal_time: 100000           # linear anneal over 100k steps
```

### Batch & Buffer
```yaml
runner: "parallel"                    # multi-process data collection
batch_size_run: 8                     # 8 parallel environments
buffer_size: 5000                     # replay buffer capacity (episodes)
batch_size: 128                       # training batch size
target_update_interval_or_tau: 200    # hard update every 200 episodes
```

### Observation Augmentation
```yaml
obs_agent_id: True                    # append one-hot agent ID to obs
obs_last_action: True                 # append previous action one-hot
obs_individual_obs: True              # per-agent observations
```

### Network Architecture
```yaml
mac: "casvd_mac"                      # CASVD multi-agent controller
agent: "gat_ns"                       # GATNSAgent
agent_output_type: "q"                # output raw Q-values
learner: "casvd_learner"              # CASVD learner
mixer: "qmix"                         # mixer type (NMixer or QMIX)
mixing_embed_dim: 32                  # mixer embedding dimension
hypernet_embed: 64                    # hypernet hidden dimension
hidden_dim: 128                       # GAT + GRU hidden dimension
n_heads: 4                            # multi-head attention heads
use_rnn: True                         # GRU recurrence
use_layer_norm: True                  # layer normalization after GAT
use_orthogonal: True                  # orthogonal weight initialization
encoder_gain: 1.0                     # init gain for encoder weights
q_head_gain: 0.1                      # init gain for Q-head (small!)
use_q_output_tanh: True               # tanh squashing on Q output
q_output_scale: 2.0                   # Q in [-2, +2]
```

### Learning
```yaml
gamma: 0.99                           # discount factor
td_lambda: 0.6                        # TD-lambda parameter
lr: 0.001                             # main learning rate
optimizer: "adam"
optimizer_epsilon: 0.0000001          # Adam epsilon (1e-7)
grad_norm_clip: 10                    # gradient norm clipping
```

### Soft Values
```yaml
use_soft_values: True                 # enable soft value targets
alpha_factor_init: 0.5                # initial alpha multiplier
alpha_factor_min: 0.0                 # minimum alpha_factor
alpha_factor_max: 1.0                 # maximum alpha_factor
alpha_floor: 0.005                    # minimum effective temperature
```

### InfoNCE Coordination Sensor
```yaml
lgdd_enabled: True                    # enable InfoNCE sensor
lgdd_lr: 0.0003                       # predictor learning rate
infonce_n_negatives: 15               # K negative samples
infonce_temperature: 0.1              # contrastive temperature tau
```

### Adaptive Alpha
```yaml
use_adaptive_alpha: True              # per-agent alpha from coordination
coord_signal_ema_tau: 0.999           # EMA smoothing (strong)
```

### Continual Learning (Optional)
```yaml
cl_enabled: False                     # disabled by default
cl_memory_buffer_size: 10000          # reservoir buffer size
cl_current_ratio: 0.8                 # 80% current, 20% memory
cl_min_memory_episodes: 128           # minimum episodes before CL kicks in
cl_teacher_ema_tau: 0.002             # teacher EMA rate
cl_distill_weight: 0.05              # KL distillation loss weight
cl_temperature: 0.1                   # softmax temperature for KL
```

---

## 12. Key Design Decisions and Rationale

### 1. Gradient Isolation for InfoNCE

**Decision**: InfoNCE never touches the encoder. Embeddings are `.detach()`-ed.

**Rationale**: If InfoNCE gradients flowed into the encoder, they would compete with Q-learning gradients. The encoder would be torn between making Q-values accurate (TD loss) and making embeddings predictive of global future (InfoNCE). These objectives can conflict — an agent might learn embeddings that are great for predicting the future but terrible for estimating Q-values. By detaching, InfoNCE acts as a read-only sensor: it observes embeddings to measure coordination but never pushes the encoder in any direction.

### 2. Q-Spread-Relative Alpha (Not Fixed Temperature)

**Decision**: Temperature `alpha = factor * (max(Q) - mean(Q))` instead of a fixed constant.

**Rationale**: A fixed temperature would need careful tuning per environment and would become invalid as Q-values evolve during training. The Q-spread-relative formulation:
- Automatically adapts to any Q-value scale
- Is invariant to uniform Q-value scaling (mathematically proven in Section 3.7)
- Creates a self-stabilizing feedback loop (Section 3.6)
- Works across different maps, agent counts, and training stages without retuning

### 3. Double-Q for Soft Values

**Decision**: Online network selects soft policy, target network evaluates Q-values.

**Rationale**: Standard soft values using a single network would overestimate because:
```
E_pi[Q(a)] with pi = softmax(Q/alpha)
```
The same Q-values that create the policy also evaluate it, leading to bias. Double-Q breaks this coupling: the online Q-values determine which actions get high probability, but the target Q-values (independently estimated) determine what those actions are actually worth.

### 4. Tanh Squashing with scale=2.0

**Decision**: `Q = tanh(Q_raw) * 2.0`, bounding Q-values to `[-2, +2]`.

**Rationale**: Unbounded Q-values can cause:
- Numerical instability in softmax computation (overflow/underflow)
- Runaway bootstrapping (Q-values grow without bound)
- Difficulty for the mixer's hypernetworks (extreme input ranges)

The `[-2, +2]` range provides sufficient dynamic range for learning while preventing these issues. The small `q_head_gain=0.1` initialization ensures Q-values start near zero, giving near-uniform initial soft policies (natural exploration without relying solely on epsilon-greedy).

### 5. Small Q-Head Initialization (gain=0.1)

**Decision**: Q-head weights initialized with orthogonal gain=0.1 while encoder uses gain=1.0.

**Rationale**: With `Q = tanh(W_q * h) * 2.0`:
- gain=0.1 means initial `W_q * h` is small → `tanh(small) ≈ small` → `Q ≈ 0`
- Near-zero Q-values → near-uniform softmax → natural exploration
- The encoder can have normal-scale features while Q-values start small
- This prevents early training instability where random large Q-values could cause the soft policy to commit to bad actions before any learning has occurred

### 6. Dual Optimizers with Different Learning Rates

**Decision**: Main network lr=0.001, InfoNCE predictor lr=0.0003.

**Rationale**: The InfoNCE predictor is a simple linear layer that converges quickly. If trained at the same rate as the main network, it would overfit to current embeddings and oscillate. The lower learning rate and complete gradient isolation ensure the coordination sensor provides stable, slowly-evolving signals that don't disrupt Q-learning.

### 7. Strong EMA Smoothing for Coordination Signals (tau=0.999)

**Decision**: Coordination signals are smoothed with EMA tau=0.999 (very slow update).

**Rationale**: The InfoNCE loss is noisy (depends on random negative sampling and batch composition). Without strong smoothing, the per-agent alpha factors would oscillate wildly, destabilizing soft value computation. The 0.999 tau means the signal changes very gradually — approximately 1000 updates to reach a new equilibrium — providing a stable coordination measure.

### 8. Entity Masking for Dead/Invisible Entities

**Decision**: Zero-feature entities are masked to `-1e9` before attention softmax.

**Rationale**: In SMACv2, dead or out-of-sight entities have all-zero features. Without masking, attention would still distribute weight to these "ghost" entities, polluting the agent's local summary with meaningless information. The `-1e9` mask ensures zero attention weight after softmax, effectively removing dead entities from the attention computation.

---

## 13. File Reference Map

| Component | File Path | Key Lines | Key Classes/Functions |
|-----------|-----------|-----------|----------------------|
| **Main Entry** | `src/main.py` | 27-38 | `my_main()` |
| **Training Loop** | `src/run/run.py` | 76-273 | `run_sequential()` |
| **CASVD Learner** | `src/learners/casvd_learner.py` | 35-485 | `CASVDLearner` |
| **CASVD Controller** | `src/controllers/casvd_controller.py` | 6-116 | `CASVDMAC` |
| **GATNSAgent** | `src/modules/agents/gat_ns_agent.py` | 118-305 | `GATNSAgent`, `LocalEntityGAT`, `TeamGATLayer` |
| **LocalEntityGAT** | `src/modules/agents/gat_ns_agent.py` | 11-43 | `LocalEntityGAT` |
| **TeamGATLayer** | `src/modules/agents/gat_ns_agent.py` | 46-71 | `TeamGATLayer` |
| **InfoNCE Predictor** | `src/modules/predictors/infonce_predictor.py` | 24-80 | `InfoNCEPredictor` |
| **NMixer** | `src/modules/mixers/nmix.py` | 8-71 | `NMixer` |
| **QMIX Mixer** | `src/modules/mixers/qmix.py` | 7-64 | `QMixer` |
| **Episode Buffer** | `src/components/episode_buffer.py` | — | `ReplayBuffer`, `EpisodeBatch` |
| **Action Selectors** | `src/components/action_selectors.py` | — | `EpsilonGreedyActionSelector` |
| **Episode Runner** | `src/runners/episode_runner.py` | 56-125 | `EpisodeRunner.run()` |
| **Parallel Runner** | `src/runners/parallel_runner.py` | 14-120 | `ParallelRunner` |
| **SMACv2 Wrapper** | `src/envs/smacv2_wrapper.py` | 38-187 | `SMACv2Wrapper` |
| **CASVD Config** | `src/config/algs/casvd.yaml` | 1-108 | YAML configuration |
| **Default Config** | `src/config/default.yaml` | — | Base hyperparameters |
| **Env Config** | `src/config/envs/sc2v2.yaml` | — | SMACv2 environment settings |
| **Map Configs** | `src/config/envs/smacv2_configs/*.yaml` | — | Per-map scenario configs |
