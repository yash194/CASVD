# CASVD: Coordination-Aware Soft Value Decomposition — Mathematical Reference

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

2. **Q-spread-relative soft values** — replaces the hard $\max$ operator in standard QMIX targets with a Boltzmann softmax whose temperature automatically adapts to the Q-value scale. This creates a self-stabilizing feedback loop that prevents value overestimation while maintaining learning signal.

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

The `_split_obs()` method (lines 289–305) parses this flat vector. Let $\mathbf{o}_{b,i} \in \mathbb{R}^{D_\text{obs}}$ denote the raw observation of agent $i$ in batch element $b$. The observation is partitioned as:

$$\mathbf{o}_{b,i} = \Big[\underbrace{\mathbf{m}_{b,i}}_{D_\text{move}} \;\Big|\; \underbrace{\mathbf{e}^1_{b,i}, \ldots, \mathbf{e}^E_{b,i}}_{E \times D_\text{enemy}} \;\Big|\; \underbrace{\mathbf{a}^1_{b,i}, \ldots, \mathbf{a}^A_{b,i}}_{A \times D_\text{ally}} \;\Big|\; \underbrace{\mathbf{p}_{b,i}}_{D_\text{own}}\Big]$$

where:
- $\mathbf{m}_{b,i} \in \mathbb{R}^{D_\text{move}}$ — movement features
- $\mathbf{e}^j_{b,i} \in \mathbb{R}^{D_\text{enemy}}$ — features of enemy $j$, for $j = 1,\ldots,E$
- $\mathbf{a}^j_{b,i} \in \mathbb{R}^{D_\text{ally}}$ — features of ally $j$, for $j = 1,\ldots,A$
- $\mathbf{p}_{b,i} \in \mathbb{R}^{D_\text{own}}$ — own-state features

Stacking across entities gives:

$$\mathbf{E}^\text{enemy}_{b,i} = \begin{bmatrix}\mathbf{e}^1_{b,i} \\ \vdots \\ \mathbf{e}^E_{b,i}\end{bmatrix} \in \mathbb{R}^{E \times D_\text{enemy}}, \qquad \mathbf{E}^\text{ally}_{b,i} = \begin{bmatrix}\mathbf{a}^1_{b,i} \\ \vdots \\ \mathbf{a}^A_{b,i}\end{bmatrix} \in \mathbb{R}^{A \times D_\text{ally}}$$

### 2.2 Entity Encoding

Each entity type is projected into a shared $d = 128$ dimensional space using separate linear encoders with weight matrices $\mathbf{W}_\text{self} \in \mathbb{R}^{(D_\text{move}+D_\text{own}) \times d}$, $\mathbf{W}_\text{enemy} \in \mathbb{R}^{D_\text{enemy} \times d}$, $\mathbf{W}_\text{ally} \in \mathbb{R}^{D_\text{ally} \times d}$:

$$\mathbf{s}_{b,i} = \text{ReLU}\!\left(\mathbf{W}_\text{self}\, \begin{bmatrix}\mathbf{m}_{b,i} \\ \mathbf{p}_{b,i}\end{bmatrix}\right) \in \mathbb{R}^{d}$$

$$\mathbf{E}^\text{enemy-enc}_{b,i,j} = \text{ReLU}\!\left(\mathbf{W}_\text{enemy}\, \mathbf{e}^j_{b,i}\right) \in \mathbb{R}^{d}, \quad j = 1,\ldots,E$$

$$\mathbf{E}^\text{ally-enc}_{b,i,j} = \text{ReLU}\!\left(\mathbf{W}_\text{ally}\, \mathbf{a}^j_{b,i}\right) \in \mathbb{R}^{d}, \quad j = 1,\ldots,A$$

An **entity mask** $\mathcal{M}_{b,i} \subseteq \{1,\ldots,1+A+E\}$ marks visible entities (entities with all-zero features are dead or out-of-sight):

$$M_{b,i,e} = \begin{cases} 1 & \text{if entity } e \text{ is visible to agent } i \\ 0 & \text{if entity } e \text{ is dead or out-of-sight} \end{cases}$$

All entity encodings are stacked into a single entity matrix:

$$\mathbf{X}_{b,i} = \left[\mathbf{s}_{b,i},\; \mathbf{E}^\text{ally-enc}_{b,i,1},\ldots,\mathbf{E}^\text{ally-enc}_{b,i,A},\; \mathbf{E}^\text{enemy-enc}_{b,i,1},\ldots,\mathbf{E}^\text{enemy-enc}_{b,i,E}\right] \in \mathbb{R}^{(1+A+E) \times d}$$

### 2.3 LocalEntityGAT — Agent-to-Entity Attention

**File**: `src/modules/agents/gat_ns_agent.py`, lines 11–43

Each agent independently attends over its own observable entities using multi-head scaled dot-product attention. This is the first layer of the two-tier attention architecture.

**Parameters**:
- Number of heads: $H = 4$
- Head dimension: $d_h = d / H = 128 / 4 = 32$
- Projection matrices: $\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V \in \mathbb{R}^{d \times d}$ (no bias), $\mathbf{W}^O \in \mathbb{R}^{d \times d}$ (with bias)

**Step 1 — Multi-head projection.**

Given query node $\mathbf{s}_{b,i} \in \mathbb{R}^d$ (agent self-embedding) and entity nodes $\mathbf{X}_{b,i} \in \mathbb{R}^{(1+A+E) \times d}$, project into $H$ heads of dimension $d_h$:

$$\mathbf{Q}^h_{b,i} = \mathbf{W}^{Q,h}\, \mathbf{s}_{b,i} \in \mathbb{R}^{d_h}, \quad h = 1,\ldots,H$$

$$\mathbf{K}^h_{b,i,e} = \mathbf{W}^{K,h}\, \mathbf{X}_{b,i,e} \in \mathbb{R}^{d_h}, \quad e = 1,\ldots,1+A+E$$

$$\mathbf{V}^h_{b,i,e} = \mathbf{W}^{V,h}\, \mathbf{X}_{b,i,e} \in \mathbb{R}^{d_h}$$

where $\mathbf{W}^{Q,h}, \mathbf{W}^{K,h}, \mathbf{W}^{V,h} \in \mathbb{R}^{d_h \times d}$ are the head-$h$ slices of the full projection matrices.

**Step 2 — Scaled dot-product attention scores.**

$$\ell^h_{b,i,e} = \frac{\mathbf{Q}^h_{b,i} \cdot \mathbf{K}^h_{b,i,e}}{\sqrt{d_h}}, \quad \forall\, h,\, e$$

**Step 3 — Mask invalid entities.**

$$\tilde{\ell}^h_{b,i,e} = \begin{cases} \ell^h_{b,i,e} & \text{if } M_{b,i,e} = 1 \\ -\infty & \text{if } M_{b,i,e} = 0 \end{cases}$$

**Step 4 — Attention weights.**

$$\alpha^h_{b,i,e} = \frac{\exp\!\left(\tilde{\ell}^h_{b,i,e}\right)}{\displaystyle\sum_{e'=1}^{1+A+E} \exp\!\left(\tilde{\ell}^h_{b,i,e'}\right)}$$

**Step 5 — Weighted aggregation.**

$$\mathbf{c}^h_{b,i} = \sum_{e=1}^{1+A+E} \alpha^h_{b,i,e}\; \mathbf{V}^h_{b,i,e} \in \mathbb{R}^{d_h}$$

**Step 6 — Concatenate heads and project.**

$$\mathbf{c}_{b,i} = \text{Concat}\!\left(\mathbf{c}^1_{b,i}, \ldots, \mathbf{c}^H_{b,i}\right) \in \mathbb{R}^{d}$$

$$\mathbf{u}_{b,i} = \mathbf{W}^O\, \mathbf{c}_{b,i} + \mathbf{b}^O \in \mathbb{R}^{d}$$

**Step 7 — Residual connection and normalisation.**

$$\boldsymbol{\ell}^{\text{loc}}_{b,i} = \mathbf{u}_{b,i} + \mathbf{s}_{b,i}$$

$$\boldsymbol{\ell}^{\text{loc}}_{b,i} \leftarrow \text{LayerNorm}\!\left(\text{ELU}\!\left(\boldsymbol{\ell}^{\text{loc}}_{b,i}\right)\right) \in \mathbb{R}^d$$

If extra input features $\boldsymbol{\xi}_{b,i} \in \mathbb{R}^{D_\text{extra}}$ exist (agent ID, last action):

$$\boldsymbol{\ell}^{\text{loc}}_{b,i} \leftarrow \boldsymbol{\ell}^{\text{loc}}_{b,i} + \text{ReLU}\!\left(\mathbf{W}_\text{extra}\, \boldsymbol{\xi}_{b,i}\right)$$

**Intuition**: Each agent builds a **local embedding** $\boldsymbol{\ell}^{\text{loc}}_{b,i}$ that summarises what it observes — nearby enemies, allies, and its own state — using attention to weight the importance of each entity.

### 2.4 TeamGATLayer — Agent-to-Agent Attention

**File**: `src/modules/agents/gat_ns_agent.py`, lines 46–71

After each agent has its local summary, all agents attend to each other to share information. This uses the classical GAT (Graph Attention Network) formulation with additive attention.

**Parameters**:
- Number of heads: $H = 4$, head dimension $d_h = d/H = 32$
- Shared linear projection: $\mathbf{W} \in \mathbb{R}^{d \times d}$ (no bias)
- Per-head attention vectors: $\mathbf{a}^h_\text{src}, \mathbf{a}^h_\text{dst} \in \mathbb{R}^{d_h}$ for $h = 1,\ldots,H$
- Activation: $\text{LeakyReLU}$ with negative slope $0.2$

**Step 1 — Project to multi-head space.**

$$\mathbf{h}^h_{b,i} = \mathbf{W}^h\, \boldsymbol{\ell}^{\text{loc}}_{b,i} \in \mathbb{R}^{d_h}, \quad h = 1,\ldots,H$$

where $\mathbf{W}^h \in \mathbb{R}^{d_h \times d}$ is the head-$h$ slice of $\mathbf{W}$.

**Step 2 — Additive GAT attention coefficients.**

For each head $h$ and each agent pair $(i, j)$:

$$e^h_{b,ij} = \text{LeakyReLU}\!\left(\mathbf{a}^h_\text{src} \cdot \mathbf{h}^h_{b,i} + \mathbf{a}^h_\text{dst} \cdot \mathbf{h}^h_{b,j}\right) \in \mathbb{R}$$

**Step 3 — Attention weights (softmax over neighbour agents $j$).**

$$\beta^h_{b,ij} = \frac{\exp\!\left(e^h_{b,ij}\right)}{\displaystyle\sum_{j'=1}^{N} \exp\!\left(e^h_{b,ij'}\right)}, \quad \sum_j \beta^h_{b,ij} = 1$$

**Step 4 — Weighted aggregation per head.**

$$\mathbf{f}^h_{b,i} = \sum_{j=1}^{N} \beta^h_{b,ij}\; \mathbf{h}^h_{b,j} \in \mathbb{R}^{d_h}$$

In matrix form: $\mathbf{F}^h_b = \boldsymbol{\beta}^h_b\, \mathbf{H}^h_b$ where $\boldsymbol{\beta}^h_b \in \mathbb{R}^{N \times N}$ and $\mathbf{H}^h_b \in \mathbb{R}^{N \times d_h}$.

**Step 5 — Concatenate heads and normalise.**

$$\boldsymbol{\ell}^{\text{team}}_{b,i} = \text{LayerNorm}\!\left(\text{ELU}\!\left(\text{Concat}\!\left(\mathbf{f}^1_{b,i}, \ldots, \mathbf{f}^H_{b,i}\right)\right)\right) \in \mathbb{R}^d$$

**Intuition**: The team GAT allows agents to form a **team-aware embedding** $\boldsymbol{\ell}^{\text{team}}_{b,i}$. Agent $i$ can learn what agents $j$ and $k$ are planning by attending to their local summaries.

### 2.5 GRU Temporal Core + Q-Head

After the two-tier attention, each agent's local and team summaries are concatenated and fed through a shared GRU for temporal reasoning.

**GRU Input Construction.**

$$\mathbf{x}^{\text{gru}}_{b,i} = \left[\boldsymbol{\ell}^{\text{loc}}_{b,i} \;\Big\|\; \boldsymbol{\ell}^{\text{team}}_{b,i}\right] \in \mathbb{R}^{2d}$$

**GRU Recurrence.** Let $\mathbf{h}^{t-1}_{b,i} \in \mathbb{R}^d$ be the hidden state from the previous timestep. Then:

$$\mathbf{z}^t_{b,i} = \sigma\!\left(\mathbf{W}_z \begin{bmatrix}\mathbf{x}^{\text{gru}}_{b,i} \\ \mathbf{h}^{t-1}_{b,i}\end{bmatrix}\right) \qquad \text{(update gate)}$$

$$\mathbf{r}^t_{b,i} = \sigma\!\left(\mathbf{W}_r \begin{bmatrix}\mathbf{x}^{\text{gru}}_{b,i} \\ \mathbf{h}^{t-1}_{b,i}\end{bmatrix}\right) \qquad \text{(reset gate)}$$

$$\tilde{\mathbf{h}}^t_{b,i} = \tanh\!\left(\mathbf{W}_h \begin{bmatrix}\mathbf{x}^{\text{gru}}_{b,i} \\ \mathbf{r}^t_{b,i} \odot \mathbf{h}^{t-1}_{b,i}\end{bmatrix}\right) \qquad \text{(candidate hidden state)}$$

$$\mathbf{h}^t_{b,i} = \left(\mathbf{1} - \mathbf{z}^t_{b,i}\right) \odot \mathbf{h}^{t-1}_{b,i} + \mathbf{z}^t_{b,i} \odot \tilde{\mathbf{h}}^t_{b,i} \in \mathbb{R}^d$$

where $\sigma(\cdot)$ is the sigmoid function and $\odot$ is element-wise multiplication.

**Q-Head with Tanh Squashing.**

$$\mathbf{q}^{\text{raw}}_{b,i} = \mathbf{W}_q\, \mathbf{h}^t_{b,i} + \mathbf{b}_q \in \mathbb{R}^{|\mathcal{A}|}$$

$$Q_{b,i,a} = \varsigma \cdot \tanh\!\left(q^{\text{raw}}_{b,i,a}\right), \quad \varsigma = 2.0 \implies Q_{b,i,a} \in [-2,\, +2]$$

**Weight initialisation**:
- Encoder matrices ($\mathbf{W}_\text{self}, \mathbf{W}_\text{ally}, \mathbf{W}_\text{enemy}, \mathbf{W}^O$): orthogonal with gain $g_\text{enc} = 1.0$
- Q-head matrix $\mathbf{W}_q$: orthogonal with gain $g_q = 0.1$ (small initial Q-values)

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

**File**: `src/learners/casvd_learner.py`, lines 182–242

In standard QMIX, targets are computed using the hard maximum:

$$V^\text{target}(s, i) = \max_{a \in \mathcal{A}_i(s)} Q^\text{target}(s, i, a)$$

CASVD replaces this with a **Boltzmann soft maximum** whose temperature is proportional to the Q-value spread, creating a self-stabilising mechanism.

### 3.1 Q-Spread Computation

For each agent $i$ at state $s$ and timestep $t$, with available action set $\mathcal{A}_i(s)$, define:

$$Q^\text{masked}(s, i, a) = \begin{cases} Q^\text{online}(s, i, a) & \text{if } a \in \mathcal{A}_i(s) \\ -\infty & \text{otherwise} \end{cases}$$

$$Q^\text{max}(s, i) = \max_{a \in \mathcal{A}_i(s)} Q^\text{online}(s, i, a)$$

$$\bar{Q}(s, i) = \frac{1}{|\mathcal{A}_i(s)|} \sum_{a \in \mathcal{A}_i(s)} Q^\text{online}(s, i, a)$$

$$\Delta Q(s, i) = \max\!\Big(Q^\text{max}(s, i) - \bar{Q}(s, i),\; \epsilon_0\Big), \quad \epsilon_0 = 10^{-6}$$

The $\epsilon_0$ floor prevents division by zero when all Q-values are equal.

### 3.2 Per-Agent Temperature Alpha

The effective temperature for agent $i$ is:

$$\alpha_i(s, t) = \max\!\Big(\phi_i \cdot \Delta Q(s, i, t),\; \alpha_\text{floor}\Big), \quad \alpha_\text{floor} = 0.005$$

where:
- $\phi_i \in [0, 1]$ is the per-agent adaptive factor driven by the coordination signal (see Section 5)
- $\alpha_\text{floor} = 0.005$ is a minimum temperature to prevent numerical issues

### 3.3 Boltzmann Soft Policy (Double-Q Style)

The soft policy is computed using the **online** network's Q-values (Double-Q: online selects, target evaluates):

$$\pi^\text{soft}(a \mid s, i) = \frac{\exp\!\left(Q^\text{online}(s, i, a) \,/\, \alpha_i(s,t)\right)}{\displaystyle\sum_{a' \in \mathcal{A}_i(s)} \exp\!\left(Q^\text{online}(s, i, a') \,/\, \alpha_i(s,t)\right)}$$

Unavailable actions are masked out before the softmax by setting them to $-10^{10}$.

### 3.4 Soft V-Value from Target Network

The soft value for each agent is the expectation of target Q-values under the soft policy:

$$V^\text{soft}(s, i) = \sum_{a \in \mathcal{A}_i(s)} \pi^\text{soft}(a \mid s, i)\; Q^\text{target}(s, i, a) = \mathbb{E}_{a \sim \pi^\text{soft}}\!\left[Q^\text{target}(s, i, a)\right]$$

This is the **Double-Q adaptation for soft values**:
- The **online** network determines the soft policy (which actions to weight)
- The **target** network provides the Q-value estimates (what those actions are worth)
- This decoupling prevents the overestimation bias that would occur if the same network both selected and evaluated

### 3.5 Mixing and TD-Lambda Returns

The per-agent soft values are mixed into a joint team value:

$$Q^\text{tot}_\text{target}(s) = \text{NMixer}\!\left(\left[V^\text{soft}(s, 1), \ldots, V^\text{soft}(s, N)\right],\; s\right)$$

Then TD($\lambda$) returns are computed via the backward recursion:

$$G_T = Q^\text{tot}_\text{target}(s_T)$$

$$G_t = r_t + \gamma (1 - d_t)\Big[\lambda\, G_{t+1} + (1 - \lambda)\, Q^\text{tot}_\text{target}(s_{t+1})\Big]$$

where $\gamma = 0.99$, $\lambda = 0.6$, and $d_t \in \{0,1\}$ is the episode termination flag.

### 3.6 Self-Stabilizing Property — Mathematical Proof

The Q-spread-relative temperature creates a **negative feedback loop** that prevents both value explosion and value collapse:

$$\text{Large } Q\text{-values} \xrightarrow{\;\Delta Q \uparrow\;} \text{Large } \alpha \xrightarrow{\;\pi^\text{soft} \text{ flattens}\;} \text{Suppressed targets} \xrightarrow{\;\text{TD update}\;} \text{Smaller } Q\text{-values}$$

$$\text{Small } Q\text{-values} \xrightarrow{\;\Delta Q \downarrow\;} \text{Small } \alpha \xrightarrow{\;\pi^\text{soft} \to \arg\max\;} \text{Hard-max targets} \xrightarrow{\;\text{TD update}\;} \text{Normal learning signal preserved}$$

**Equilibrium condition**: The system reaches a natural equilibrium where:

$$\alpha^*_i = \phi_i \cdot \Delta Q^*(s, i) \quad \text{such that} \quad \mathbb{E}_{a \sim \pi^\text{soft}_{\alpha^*}}\!\left[Q^\text{target}(s, i, a)\right] \approx \max_a Q^\text{target}(s, i, a) - \epsilon$$

for small $\epsilon > 0$ — the soft target is close to the hard-max target without instability.

### 3.7 Scale-Invariance Property

The soft value formulation is **invariant to uniform scaling of Q-values**:

**Theorem**: Let $Q'(s, i, a) = c \cdot Q(s, i, a)$ for all $s, i, a$ and constant $c > 0$. Then $\pi^\text{soft}_{Q'} = \pi^\text{soft}_Q$.

**Proof**:

$$\Delta Q'(s, i) = \max_{a} Q'(s,i,a) - \bar{Q}'(s,i) = c \cdot \max_a Q(s,i,a) - c \cdot \bar{Q}(s,i) = c \cdot \Delta Q(s,i)$$

$$\alpha'_i(s,t) = \phi_i \cdot \Delta Q'(s,i) = \phi_i \cdot c \cdot \Delta Q(s,i) = c \cdot \alpha_i(s,t)$$

Therefore:

$$\frac{Q'(s,i,a)}{\alpha'_i(s,t)} = \frac{c \cdot Q(s,i,a)}{c \cdot \alpha_i(s,t)} = \frac{Q(s,i,a)}{\alpha_i(s,t)}$$

Hence $\pi^\text{soft}_{Q'} = \text{softmax}(Q'/\alpha') = \text{softmax}(Q/\alpha) = \pi^\text{soft}_Q$. $\blacksquare$

This means the algorithm is robust across different training stages, reward scales, and numbers of agents without retuning.

---

## 4. InfoNCE Coordination Sensor

**File**: `src/modules/predictors/infonce_predictor.py`

### 4.1 Motivation

In multi-agent systems, some agents may be well-coordinated with the team (their local information is predictive of the team's future) while others may be acting independently or confused. CASVD uses InfoNCE contrastive learning to **measure** this coordination without **interfering** with Q-value learning.

The key question InfoNCE answers: **"How well does agent $i$'s local embedding predict what the global team state will be at the next timestep?"**

- If well: agent $i$ is coordinated (its local decisions align with team outcomes)
- If poorly: agent $i$ is uncoordinated (its local view does not capture team dynamics)

### 4.2 Architecture

The InfoNCE predictor is minimal — a single linear projection:

$$\mathbf{W}_\text{pred} \in \mathbb{R}^{d \times d}, \quad d = 128 \qquad \text{(no bias, orthogonal initialisation)}$$

Temperature parameter: $\tau = 0.1$ (controls sharpness of the contrastive distribution).

### 4.3 InfoNCE Loss Computation

**File**: `src/modules/predictors/infonce_predictor.py`, lines 34–80

**Inputs** (all detached from encoder — no gradient flows back to GAT):

- $\mathbf{h}_i \in \mathbb{R}^{B \times N \times d}$: agent local embeddings $\boldsymbol{\ell}^\text{loc}_{b,i}$ at time $t$
- $\mathbf{g}^+ \in \mathbb{R}^{B \times d}$: real global team state at $t+1$
- $\mathbf{g}^-_k \in \mathbb{R}^{B \times d}$: fake global states from $K = 15$ random timesteps

The global team state at each timestep $t$ is defined as the mean-pooled local summary:

$$\mathbf{g}(t) = \frac{1}{N}\sum_{i=1}^{N} \boldsymbol{\ell}^\text{loc}_{b,i}(t) \in \mathbb{R}^d$$

**Step 1 — Project local embeddings.**

$$\hat{\mathbf{h}}_{b,i} = \mathbf{W}_\text{pred}\, \mathbf{h}_{b,i} \in \mathbb{R}^d$$

**Step 2 — L2 normalise all vectors to the unit hypersphere.**

$$\tilde{\mathbf{h}}_{b,i} = \frac{\hat{\mathbf{h}}_{b,i}}{\|\hat{\mathbf{h}}_{b,i}\|_2}, \qquad \tilde{\mathbf{g}}^+ = \frac{\mathbf{g}^+}{\|\mathbf{g}^+\|_2}, \qquad \tilde{\mathbf{g}}^-_k = \frac{\mathbf{g}^-_k}{\|\mathbf{g}^-_k\|_2}$$

L2 normalisation ensures all similarity scores lie in $[-1/\tau,\; +1/\tau]$ regardless of embedding magnitude, preventing the loss from being dominated by vector norms.

**Step 3 — Compute scaled cosine similarity scores.**

$$s^+_{b,i} = \frac{\tilde{\mathbf{h}}_{b,i} \cdot \tilde{\mathbf{g}}^+_b}{\tau} \in \mathbb{R} \qquad \text{(positive pair score)}$$

$$s^{-,k}_{b,i} = \frac{\tilde{\mathbf{h}}_{b,i} \cdot \tilde{\mathbf{g}}^{-,k}_b}{\tau} \in \mathbb{R}, \quad k = 1,\ldots,K \qquad \text{(negative pair scores)}$$

**Step 4 — Concatenate into logit vector.**

$$\boldsymbol{\ell}^\text{NCE}_{b,i} = \left[s^+_{b,i},\; s^{-,1}_{b,i},\; \ldots,\; s^{-,K}_{b,i}\right] \in \mathbb{R}^{K+1}$$

The positive sample is always placed at index $0$.

**Step 5 — InfoNCE cross-entropy loss.**

$$\mathcal{L}^\text{NCE}_{b,i} = -\log \frac{\exp\!\left(s^+_{b,i}\right)}{\exp\!\left(s^+_{b,i}\right) + \displaystyle\sum_{k=1}^{K} \exp\!\left(s^{-,k}_{b,i}\right)} = \text{CrossEntropy}\!\left(\boldsymbol{\ell}^\text{NCE}_{b,i},\; y=0\right)$$

**Loss range**:

$$\mathcal{L}^\text{NCE}_{b,i} \in \left[0,\; \log(K+1)\right] = \left[0,\; \log 16 \approx 2.77\right]$$

- Minimum $0$: the model perfectly identifies the positive sample
- Maximum $\log(K+1)$: random guessing (uniform over $K+1$ options)

### 4.4 Coordination Signal Derivation

**File**: `src/learners/casvd_learner.py`, lines 337–344

**Step 1 — Per-timestep InfoNCE losses.** For each $t \in \{0, \ldots, T-2\}$:

$$\mathbf{h}_i(t) = \boldsymbol{\ell}^\text{loc}(t)\big|_\text{detached}, \qquad \mathbf{g}^+(t) = \mathbf{g}(t+1)\big|_\text{detached}$$

$$\mathbf{g}^-_k(t) = \mathbf{g}(t_k)\big|_\text{detached}, \quad t_k \sim \text{Uniform}\!\left(\{0,\ldots,T-1\} \setminus \{t+1\}\right)$$

$$\mathcal{L}^\text{NCE}_t = \text{InfoNCEPredictor}\!\left(\mathbf{h}_i(t),\; \mathbf{g}^+(t),\; \left[\mathbf{g}^-_k(t)\right]_{k=1}^K\right) \in \mathbb{R}^{B \times N}$$

**Step 2 — Time-averaged per-agent loss** (masked for variable-length episodes).

Let $m_{b,t} \in \{0,1\}$ be the validity mask. Then:

$$\bar{\mathcal{L}}^\text{NCE}_{b,i} = \frac{\displaystyle\sum_{t=0}^{T-2} m_{b,t}\; \mathcal{L}^\text{NCE}_{t,b,i}}{\displaystyle\sum_{t=0}^{T-2} m_{b,t}} \in \mathbb{R}^{B \times N}$$

**Step 3 — Coordination signal** (normalised and batch-averaged).

$$\hat{\sigma}^{(n)}_i = \text{clamp}\!\left(\frac{\displaystyle\frac{1}{B}\sum_{b=1}^B \bar{\mathcal{L}}^\text{NCE}_{b,i}}{\log(K+1)},\; 0,\; 1\right) \in [0, 1], \quad i = 1,\ldots,N$$

**Step 4 — Exponential Moving Average (EMA) smoothing.**

$$\sigma_i \leftarrow \tau_\text{ema}\; \sigma_i + (1 - \tau_\text{ema})\; \hat{\sigma}^{(n)}_i, \qquad \tau_\text{ema} = 0.999$$

**Interpretation**:

$$\sigma_i \approx 0 \implies \text{agent } i \text{ perfectly predicts global future} \implies \text{well-coordinated}$$

$$\sigma_i \approx 1 \implies \text{agent } i \text{ cannot predict global future} \implies \text{poorly-coordinated}$$

### 4.5 Gradient Isolation — Critical Design Choice

**This is one of the most important design decisions in CASVD.**

The InfoNCE loss trains **only** the predictor weight matrix $\mathbf{W}_\text{pred}$. The encoder embeddings are detached before being passed to the predictor:

$$\hat{\mathbf{h}}_{b,i} = \mathbf{W}_\text{pred}\, \underbrace{\boldsymbol{\ell}^\text{loc}_{b,i}\big|_\text{stop\_grad}}_{\text{no gradient here}}$$

Two completely separate backward passes enforce this:

$$\text{Pass 1:} \quad \nabla_{\mathbf{W}_\text{pred}} \mathcal{L}^\text{NCE} \quad \xrightarrow{\;\text{lgdd\_optimizer (lr}=3\times10^{-4}\text{)}\;} \quad \mathbf{W}_\text{pred} \;\text{only}$$

$$\text{Pass 2:} \quad \nabla_{\theta_\text{main}} \mathcal{L}^\text{TD} \quad \xrightarrow{\;\text{main\_optimizer (lr}=10^{-3}\text{)}\;} \quad \mathbf{W}_\text{enc},\, \mathbf{W}_\text{GRU},\, \mathbf{W}_q,\, \theta_\text{mixer}$$

**Why this matters**: If InfoNCE gradients flowed into the encoder, they would compete with Q-value learning gradients. The encoder would be torn between making Q-values accurate (TD loss) and making embeddings predictive of global future (InfoNCE). These objectives can conflict. By detaching, the InfoNCE predictor acts as a **read-only sensor** — it observes the encoder's embeddings to measure coordination, but never pushes the encoder in any direction.

---

## 5. Per-Agent Adaptive Alpha

**File**: `src/learners/casvd_learner.py`, lines 206–214

### 5.1 Mapping Coordination Signal to Alpha Factor

The coordination signal $\sigma_i$ from InfoNCE is mapped linearly to a per-agent alpha factor $\phi_i$:

$$\phi_i = \alpha_\text{min} + (\alpha_\text{max} - \alpha_\text{min})\; \sigma_i = 0.0 + (1.0 - 0.0)\; \sigma_i = \sigma_i$$

This alpha factor then scales the Q-spread to produce the effective temperature:

$$\alpha^\text{per}_i(s, t) = \phi_i \cdot \Delta Q(s, i, t)$$

$$\alpha^\text{eff}_i(s, t) = \max\!\left(\alpha^\text{per}_i(s, t),\; \alpha_\text{floor}\right), \quad \alpha_\text{floor} = 0.005$$

The complete chain from InfoNCE to soft value temperature:

$$\mathcal{L}^\text{NCE}_{b,i}(t) \xrightarrow{\;\text{normalise, average, clamp}\;} \hat{\sigma}^{(n)}_i \xrightarrow{\;\text{EMA}(\tau=0.999)\;} \sigma_i \xrightarrow{\;\phi_i = \sigma_i\;} \phi_i \xrightarrow{\;\phi_i \cdot \Delta Q_i(s,t)\;} \alpha^\text{per}_i \xrightarrow{\;\max(\cdot,\, 0.005)\;} \alpha^\text{eff}_i$$

$$\pi^\text{soft}(a \mid s, i) = \text{softmax}\!\left(\frac{Q^\text{online}(s,i,\cdot)}{\alpha^\text{eff}_i(s,t)}\right)$$

### 5.2 Behavioral Interpretation

| Agent State | $\sigma_i$ | $\phi_i$ | $\Delta Q$ (example) | $\alpha^\text{eff}_i$ | Softmax behaviour |
|-------------|-----------|---------|----------------------|----------------------|-------------------|
| Well-coordinated | $\approx 0.0$ | $\approx 0.0$ | $0.04$ | $0.005$ (floor) | Near hard-max (exploit) |
| Moderately coordinated | $\approx 0.3$ | $\approx 0.3$ | $0.04$ | $0.012$ | Slightly soft |
| Poorly coordinated | $\approx 0.8$ | $\approx 0.8$ | $0.04$ | $0.032$ | Quite soft (explore) |
| Completely uncoordinated | $\approx 1.0$ | $\approx 1.0$ | $0.04$ | $0.040$ | Very soft (maximum exploration) |

**Intuition**:
- **Well-coordinated agents** ($\sigma_i \approx 0$) get near-zero $\alpha$, meaning their soft targets approach the hard maximum. These agents already understand the team dynamics, so they should exploit their knowledge greedily.
- **Poorly-coordinated agents** ($\sigma_i \approx 1$) get higher $\alpha$, meaning their soft targets are more diffuse. These agents are confused about team dynamics, so giving them softer targets provides the mixer with more diverse Q-values to learn from, and prevents premature convergence to a suboptimal joint policy.

---

## 6. NMixer — Value Decomposition Network

**File**: `src/modules/mixers/nmix.py`

### 6.1 Architecture

The NMixer is a state-conditioned two-layer hypernetwork that combines per-agent Q-values into a joint $Q^\text{tot}$ while maintaining monotonicity. Let $e = 32$ be the embedding dimension and $e_h = 64$ the hypernet hidden dimension.

**Hypernetworks** (generate weights and biases from global state $\mathbf{s} \in \mathbb{R}^{D_s}$):

$$\mathbf{w}_1(\mathbf{s}) = f_{w_1}(\mathbf{s}) \in \mathbb{R}^{N \times e}, \qquad f_{w_1}: \mathbb{R}^{D_s} \to \mathbb{R}^{Ne} \;\text{(MLP)}$$

$$\mathbf{b}_1(\mathbf{s}) = f_{b_1}(\mathbf{s}) \in \mathbb{R}^{e}, \qquad f_{b_1}: \mathbb{R}^{D_s} \to \mathbb{R}^{e} \;\text{(Linear)}$$

$$\mathbf{w}_2(\mathbf{s}) = f_{w_2}(\mathbf{s}) \in \mathbb{R}^{e \times 1}, \qquad f_{w_2}: \mathbb{R}^{D_s} \to \mathbb{R}^{e} \;\text{(MLP)}$$

$$b_2(\mathbf{s}) = f_{b_2}(\mathbf{s}) \in \mathbb{R}, \qquad f_{b_2}: \mathbb{R}^{D_s} \to \mathbb{R} \;\text{(2-layer MLP with ReLU)}$$

### 6.2 Forward Pass — Mathematical Formulation

Given per-agent Q-values $\mathbf{q} = [Q_1, \ldots, Q_N]^\top \in \mathbb{R}^N$ and global state $\mathbf{s}$:

**Layer 1** (with positivity constraint):

$$\tilde{\mathbf{w}}_1(\mathbf{s}) = \left|\mathbf{w}_1(\mathbf{s})\right| \in \mathbb{R}^{N \times e}_{\geq 0}$$

$$\mathbf{h}(\mathbf{s}, \mathbf{q}) = \text{ELU}\!\left(\tilde{\mathbf{w}}_1(\mathbf{s})^\top \mathbf{q} + \mathbf{b}_1(\mathbf{s})\right) \in \mathbb{R}^{e}$$

**Layer 2** (with positivity constraint):

$$\tilde{\mathbf{w}}_2(\mathbf{s}) = \left|\mathbf{w}_2(\mathbf{s})\right| \in \mathbb{R}^{e \times 1}_{\geq 0}$$

$$Q^\text{tot}(\mathbf{s}, \mathbf{q}) = \tilde{\mathbf{w}}_2(\mathbf{s})^\top \mathbf{h}(\mathbf{s}, \mathbf{q}) + b_2(\mathbf{s}) \in \mathbb{R}$$

**Full mathematical form**:

$$\boxed{Q^\text{tot}(\mathbf{s}, \mathbf{q}) = b_2(\mathbf{s}) + \left|\mathbf{w}_2(\mathbf{s})\right|^\top \text{ELU}\!\left(\left|\mathbf{w}_1(\mathbf{s})\right|^\top \mathbf{q} + \mathbf{b}_1(\mathbf{s})\right)}$$

where:
- $|\mathbf{w}_1(\mathbf{s})| \in \mathbb{R}^{N \times e}_{\geq 0}$ are non-negative first-layer weights
- $|\mathbf{w}_2(\mathbf{s})| \in \mathbb{R}^{e}_{\geq 0}$ are non-negative second-layer weights
- $\mathbf{b}_1(\mathbf{s}) \in \mathbb{R}^{e}$ is unconstrained first-layer bias
- $b_2(\mathbf{s}) \in \mathbb{R}$ is unconstrained second-layer bias (replaces the separate $V(s)$ in standard QMIX)

### 6.2 Monotonicity Guarantee

Since $|\mathbf{w}_1(\mathbf{s})| \geq 0$ and $|\mathbf{w}_2(\mathbf{s})| \geq 0$, and $\text{ELU}$ is monotone, we have:

$$\frac{\partial Q^\text{tot}}{\partial Q_i} = \left[\tilde{\mathbf{w}}_2(\mathbf{s})^\top \cdot \text{ELU}'\!\left(\tilde{\mathbf{w}}_1(\mathbf{s})^\top \mathbf{q} + \mathbf{b}_1(\mathbf{s})\right) \odot \tilde{w}_{1,i}(\mathbf{s})\right] \geq 0 \quad \forall\, i$$

This is the **Individual-Global-Max (IGM)** principle: each agent can greedily maximise its own Q-value during decentralised execution, and $Q^\text{tot}$ is guaranteed to improve.

The positivity function options are:
- $\text{abs}$: $|\mathbf{w}| = \left|\mathbf{w}\right|$ (default in CASVD)
- $\text{softplus}$: $|\mathbf{w}| = \log(1 + e^\mathbf{w})$
- $\text{quadratic}$: $|\mathbf{w}| = \mathbf{w}^2$

### 6.3 Difference from Standard QMIX

| Feature | Standard QMIX | NMixer |
|---------|--------------|--------|
| Bias term | $V(s)$ separate network | State-dependent $b_2(s)$ via hypernetwork |
| Layer 2 bias | $V(s) = \text{MLP}(s)$ | $f_{b_2}(s)$ via same hypernetwork pattern |
| Weight constraint | $\text{abs}()$ | Configurable: $\text{abs}$, $\text{softplus}$, $\text{quadratic}$ |
| Architecture | 2-layer hypernet for weights | Same pattern, slightly different parameterisation |

The key structural difference is that NMixer uses $f_{b_2}(s)$ instead of a separate value function $V(s)$, making the architecture more uniform.

---

## 7. CASVD Controller

**File**: `src/controllers/casvd_controller.py`

### 7.1 Input Construction

The CASVD controller builds agent inputs by concatenating the following for each agent $i$ at timestep $t$:

$$\mathbf{x}_{b,i,t} = \begin{cases} \left[\mathbf{o}_{b,i,t} \;\|\; \mathbf{0}_{|\mathcal{A}|} \;\|\; \mathbf{e}_i\right] & \text{if } t = 0 \\[6pt] \left[\mathbf{o}_{b,i,t} \;\|\; \mathbf{a}^{\text{onehot}}_{b,i,t-1} \;\|\; \mathbf{e}_i\right] & \text{if } t > 0 \end{cases}$$

where:
- $\mathbf{o}_{b,i,t} \in \mathbb{R}^{D_\text{obs}}$ — current observation
- $\mathbf{a}^{\text{onehot}}_{b,i,t-1} \in \{0,1\}^{|\mathcal{A}|}$ — one-hot encoding of the previous action
- $\mathbf{e}_i \in \{0,1\}^N$ — one-hot agent ID (enables agent-specific behaviour under parameter sharing)

$$\mathbf{x}_{b,i,t} \in \mathbb{R}^{D_\text{obs} + |\mathcal{A}| + N}$$

### 7.2 Action Selection

**During data collection** ($\varepsilon$-greedy with linear annealing):

$$\varepsilon(t) = \max\!\left(\varepsilon_\text{min},\; \varepsilon_0 - \frac{t}{T_\text{anneal}}\right) = \max\!\left(0.05,\; 1.0 - \frac{t}{100{,}000}\right)$$

$$a_{b,i,t} = \begin{cases} \text{Uniform}(\mathcal{A}_i(s)) & \text{with probability } \varepsilon(t) \\[4pt] \displaystyle\arg\max_{a \in \mathcal{A}_i(s)} Q_{b,i,a} & \text{with probability } 1 - \varepsilon(t) \end{cases}$$

**During evaluation** (greedy, $\varepsilon = 0$):

$$a_{b,i,t} = \arg\max_{a \in \mathcal{A}_i(s)} Q_{b,i,a}$$

### 7.3 Differences from BasicMAC

1. **Raw Q-value output**: CASVD controller returns raw, unmasked Q-values. Masking for unavailable actions is handled separately by the learner and action selector. This enables clean softmax computation in the soft value formulation.

2. **Latent access**: The `forward_with_latents()` method also returns encoder latents $\left\{\boldsymbol{\ell}^\text{loc}_{b,i},\, \boldsymbol{\ell}^\text{team}_{b,i}\right\}$ needed for the InfoNCE coordination sensor.

3. **No internal masking/softmax**: Unlike BasicMAC which applies softmax for `pi_logits` output type, CASVD controller outputs raw Q-values and lets the learner handle all transformations.

---

## 8. Complete Training Flow

### 8.1 Episode Collection

The ParallelRunner spawns $P = 8$ parallel StarCraft II environments. For each environment simultaneously:

$$\text{env.reset()} \implies \text{SMACv2 samples team composition from config}$$

For $t = 0, 1, \ldots, T_\text{limit}$:

$$s_t = \text{env.get\_state}(), \quad \mathbf{o}_{i,t} = \text{env.get\_obs}(), \quad \mathcal{A}_i(s_t) = \text{env.get\_avail\_actions}()$$

$$\mathbf{Q}_{i,t} = \text{GATNSAgent}\!\left(\mathbf{x}_{i,t},\, \mathbf{h}^{t-1}_i\right), \quad a_{i,t} = \varepsilon\text{-greedy}\!\left(\mathbf{Q}_{i,t},\, \mathcal{A}_i(s_t)\right)$$

$$r_t,\, d_t = \text{env.step}\!\left([a_{1,t}, \ldots, a_{N,t}]\right)$$

Transitions $\left(s_t, \{\mathbf{o}_{i,t}\}, \{a_{i,t}\}, r_t, d_t\right)$ are stored and inserted into $\mathcal{B}$ (ReplayBuffer, capacity $= 5{,}000$).

### 8.2 Training Step — Full Sequential Walkthrough

**File**: `src/learners/casvd_learner.py`, lines 128–436

---

**STEP 1: BATCH EXTRACTION** (lines 129–143)

Sample $B = 128$ episodes from the replay buffer $\mathcal{B}$:

$$\mathcal{D} = \left\{(s^b_t,\, \mathbf{o}^b_t,\, \mathbf{a}^b_t,\, r^b_t,\, d^b_t,\, m^b_t)\right\}_{b=1,\, t=0}^{B,\, T}$$

Tensor shapes:
- $\mathbf{r} \in \mathbb{R}^{B \times (T-1) \times 1}$, $\;\mathbf{a} \in \mathbb{Z}^{B \times (T-1) \times N \times 1}$, $\;\mathbf{d} \in \{0,1\}^{B \times (T-1) \times 1}$, $\;\mathbf{m} \in \{0,1\}^{B \times (T-1) \times 1}$

---

**STEP 2: ONLINE NETWORK FORWARD PASS** (lines 148–160)

Initialise hidden states: $\mathbf{h}^0_{b,i} = \mathbf{0} \in \mathbb{R}^d$ for all $b, i$.

For $t = 0, 1, \ldots, T$:

$$\mathbf{x}^b_t = \text{build\_inputs}\!\left(\mathcal{D},\, t\right) \in \mathbb{R}^{B \times N \times (D_\text{obs}+|\mathcal{A}|+N)}$$

$$Q^b_{i,t,\cdot},\; \boldsymbol{\ell}^{b,\text{loc}}_{i,t},\; \boldsymbol{\ell}^{b,\text{team}}_{i,t} = \text{GATNSAgent}^{\text{online}}\!\left(\mathbf{x}^b_t,\, \mathbf{h}^{t-1}_{b,i}\right)$$

Stack: $\mathbf{Q}^\text{online} \in \mathbb{R}^{B \times T \times N \times |\mathcal{A}|}$, all latents saved for InfoNCE.

---

**STEP 3: TARGET NETWORK FORWARD PASS** (lines 162–176)

Same procedure with frozen target weights:

$$Q^b_{i,t,\cdot} = \text{GATNSAgent}^{\text{target}}\!\left(\mathbf{x}^b_t,\, \tilde{\mathbf{h}}^{t-1}_{b,i}\right)$$

Stack: $\mathbf{Q}^\text{target} \in \mathbb{R}^{B \times T \times N \times |\mathcal{A}|}$.

---

**STEP 4: SOFT VALUE TARGET COMPUTATION** (lines 182–242)

**4a — Q-spread:**

$$\tilde{Q}^\text{online}_{b,t,i,a} = \begin{cases} Q^\text{online}_{b,t,i,a} & \text{if } \text{avail}_{b,t,i,a} = 1 \\ -10^{10} & \text{otherwise} \end{cases}$$

$$Q^\text{max}_{b,t,i} = \max_{a}\, \tilde{Q}^\text{online}_{b,t,i,a}, \qquad \bar{Q}_{b,t,i} = \frac{\displaystyle\sum_a \tilde{Q}^\text{online}_{b,t,i,a} \cdot \text{avail}_{b,t,i,a}}{\displaystyle\sum_a \text{avail}_{b,t,i,a}}$$

$$\Delta Q_{b,t,i} = \max\!\left(Q^\text{max}_{b,t,i} - \bar{Q}_{b,t,i},\; 10^{-6}\right)$$

**4b — Per-agent adaptive alpha:**

$$\phi_i = \alpha_\text{min} + (\alpha_\text{max} - \alpha_\text{min})\, \sigma_i, \qquad \alpha_\text{min} = 0.0,\; \alpha_\text{max} = 1.0$$

$$\alpha_{b,t,i} = \max\!\left(\phi_i \cdot \Delta Q_{b,t,i},\; 0.005\right)$$

**4c — Soft policy from online network:**

$$\pi^\text{soft}_{b,t,i,a} = \frac{\exp\!\left(\tilde{Q}^\text{online}_{b,t,i,a}\, /\, \alpha_{b,t,i}\right)}{\displaystyle\sum_{a'} \exp\!\left(\tilde{Q}^\text{online}_{b,t,i,a'}\, /\, \alpha_{b,t,i}\right)}$$

**4d — Soft V-value from target network:**

$$V^\text{soft}_{b,t,i} = \sum_a \pi^\text{soft}_{b,t,i,a} \cdot Q^\text{target}_{b,t,i,a}$$

**4e — Mix through NMixer:**

$$Q^\text{tot,target}_{b,t} = \text{NMixer}\!\left(\mathbf{V}^\text{soft}_{b,t},\; \mathbf{s}_{b,t}\right) \in \mathbb{R}$$

**4f — TD-$\lambda$ returns:**

$$G^b_T = Q^\text{tot,target}_{b,T}$$

$$G^b_t = r^b_t + \gamma(1 - d^b_t)\Big[\lambda\, G^b_{t+1} + (1-\lambda)\, Q^\text{tot,target}_{b,t+1}\Big], \quad t = T-1, \ldots, 0$$

---

**STEP 5: MIX CHOSEN Q-VALUES** (lines 267–270)

$$\bar{Q}^\text{chosen}_{b,t,i} = Q^\text{online}_{b,t,i,a^b_{i,t}}, \quad t = 0, \ldots, T-2$$

$$Q^\text{tot,online}_{b,t} = \text{NMixer}\!\left(\bar{\mathbf{Q}}^\text{chosen}_{b,t},\; \mathbf{s}_{b,t}\right) \in \mathbb{R}$$

---

**STEP 6: TD LOSS** (lines 275–277)

$$\delta_{b,t} = Q^\text{tot,online}_{b,t} - G^b_t \qquad \text{(TD error)}$$

$$\mathcal{L}^\text{TD} = \frac{\displaystyle\sum_{b,t} \frac{1}{2}\, \delta^2_{b,t} \cdot m_{b,t}}{\displaystyle\sum_{b,t} m_{b,t}}$$

---

**STEP 7: InfoNCE COORDINATION SENSOR** (lines 290–351)

**7a — Collect embeddings (detached):**

$$\mathbf{H}(t) = \left[\boldsymbol{\ell}^{b,\text{loc}}_{i,t}\big|_\text{sg}\right]_{b,i} \in \mathbb{R}^{T \times B \times N \times d}, \qquad \text{sg} = \text{stop\_gradient}$$

**7b — Global team state:**

$$\mathbf{g}(t) = \frac{1}{N}\sum_{i=1}^N \mathbf{H}(t)_{:,i,:} \in \mathbb{R}^{T \times B \times d}$$

**7c — Per-timestep InfoNCE:**

For $t = 0, \ldots, T-2$:

$$\mathbf{g}^+(t) = \mathbf{g}(t+1), \qquad \mathbf{g}^-_k(t) = \mathbf{g}(t_k),\; t_k \sim \{0,\ldots,T-1\} \setminus \{t+1\}$$

$$\mathcal{L}^\text{NCE}_{b,i}(t) = -\log \frac{\exp\!\left(\tilde{\mathbf{h}}_{b,i,t} \cdot \tilde{\mathbf{g}}^+(t)_b \,/\, \tau\right)}{\exp\!\left(\tilde{\mathbf{h}}_{b,i,t} \cdot \tilde{\mathbf{g}}^+(t)_b \,/\, \tau\right) + \displaystyle\sum_{k=1}^K \exp\!\left(\tilde{\mathbf{h}}_{b,i,t} \cdot \tilde{\mathbf{g}}^-_k(t)_b \,/\, \tau\right)}$$

**7d — Time-average:**

$$\bar{\mathcal{L}}^\text{NCE}_{b,i} = \frac{\displaystyle\sum_{t=0}^{T-2} m_{b,t}\; \mathcal{L}^\text{NCE}_{b,i}(t)}{\displaystyle\sum_{t=0}^{T-2} m_{b,t}} \in \mathbb{R}^{B \times N}$$

**7e — InfoNCE loss (scalar):**

$$\mathcal{L}^\text{InfoNCE} = \frac{1}{BN}\sum_{b,i} \bar{\mathcal{L}}^\text{NCE}_{b,i}$$

**7f — Update coordination signals:**

$$\hat{\sigma}_i = \text{clamp}\!\left(\frac{\displaystyle\frac{1}{B}\sum_b \bar{\mathcal{L}}^\text{NCE}_{b,i}}{\log(K+1)},\; 0,\; 1\right), \qquad \sigma_i \leftarrow 0.999\,\sigma_i + 0.001\,\hat{\sigma}_i$$

**7g — Update scalar alpha (for logging):**

$$\bar{\phi} = \alpha_\text{min} + (\alpha_\text{max} - \alpha_\text{min})\cdot \frac{1}{N}\sum_{i=1}^N \sigma_i$$

### 8.3 Dual Backward Passes

**File**: `src/learners/casvd_learner.py`, lines 368–378

The two losses are backpropagated through completely separate computational graphs.

**Backward Pass 1 — InfoNCE predictor only:**

$$\nabla_{\mathbf{W}_\text{pred}} \mathcal{L}^\text{InfoNCE} \xrightarrow{\;\text{clip}\,\|\cdot\|_2 \leq 10\;} \mathbf{W}_\text{pred} \leftarrow \mathbf{W}_\text{pred} - \eta_\text{lgdd}\, \nabla_{\mathbf{W}_\text{pred}}\mathcal{L}^\text{InfoNCE}, \quad \eta_\text{lgdd} = 3 \times 10^{-4}$$

Parameters updated: $\mathbf{W}_\text{pred}$ only. All encoder embeddings were $\text{stop\_gradient}$-ed.

**Backward Pass 2 — Main network:**

$$\nabla_{\Theta} \mathcal{L}^\text{TD} \xrightarrow{\;\text{clip}\,\|\cdot\|_2 \leq 10\;} \Theta \leftarrow \Theta - \eta_\text{main}\, \nabla_\Theta \mathcal{L}^\text{TD}, \quad \eta_\text{main} = 10^{-3}$$

where $\Theta = \left\{\mathbf{W}_q,\, \mathbf{W}_\text{GRU},\, \mathbf{W}^\text{TeamGAT},\, \mathbf{W}^Q,\mathbf{W}^K,\mathbf{W}^V,\mathbf{W}^O,\, \mathbf{W}_\text{enc},\, \theta_\text{NMixer}\right\}$.

### 8.4 Target Network Update Strategy

**File**: `src/learners/casvd_learner.py`, lines 381–393

**Hard update** (every $T_\text{target} = 200$ episodes):

$$\text{if}\;\; (n_\text{ep} - n^\text{last}_\text{target}) \geq T_\text{target}: \quad \tilde{\Theta} \leftarrow \Theta, \quad \tilde{\theta}_\text{mixer} \leftarrow \theta_\text{mixer}$$

**Soft update** (alternative, if $\tau_\text{target} < 1$):

$$\tilde{\theta}_j \leftarrow (1 - \tau_\text{target})\, \tilde{\theta}_j + \tau_\text{target}\, \theta_j \quad \forall\, j$$

**Teacher MAC EMA update** (if continual learning enabled):

$$\theta^\text{teacher}_j \leftarrow (1 - \tau_\text{cl})\, \theta^\text{teacher}_j + \tau_\text{cl}\, \theta_j, \quad \tau_\text{cl} = 0.002$$

---

## 9. All Loss Functions

### Primary: TD Loss

$$\mathcal{L}^\text{TD} = \frac{1}{N_\text{valid}} \sum_{b=1}^{B}\sum_{t=0}^{T-2} \frac{1}{2} \left(Q^\text{tot,online}_{b,t} - G^b_t\right)^2 \cdot m_{b,t}$$

where $N_\text{valid} = \sum_{b,t} m_{b,t}$ and $G^b_t$ is the TD($\lambda$) target from Section 3.5.

**Trains**: Entire encoder ($\mathbf{W}^Q, \mathbf{W}^K, \mathbf{W}^V, \mathbf{W}^O, \mathbf{W}_\text{enc}$) + GRU + Q-head + NMixer  
**Optimizer**: Adam with $\eta = 10^{-3}$, $\epsilon = 10^{-7}$

### Secondary: InfoNCE Loss

$$\mathcal{L}^\text{InfoNCE} = \frac{1}{BN} \sum_{b=1}^{B}\sum_{i=1}^{N} \bar{\mathcal{L}}^\text{NCE}_{b,i} = \frac{-1}{BN} \sum_{b,i} \log \frac{\exp\!\left(s^+_{b,i}/\tau\right)}{\exp\!\left(s^+_{b,i}/\tau\right) + \displaystyle\sum_{k=1}^K \exp\!\left(s^{-,k}_{b,i}/\tau\right)}$$

where $s^+_{b,i} = \tilde{\mathbf{h}}_{b,i} \cdot \tilde{\mathbf{g}}^+_b$ and $s^{-,k}_{b,i} = \tilde{\mathbf{h}}_{b,i} \cdot \tilde{\mathbf{g}}^{-,k}_b$, with $\tau = 0.1$, $K = 15$.

**Trains**: Only $\mathbf{W}_\text{pred}$ (single linear layer)  
**Optimizer**: Adam with $\eta = 3 \times 10^{-4}$, $\epsilon = 10^{-7}$

### Optional: Continual Learning Distillation Loss

$$\boldsymbol{\pi}^\text{teacher} = \text{softmax}\!\left(\frac{\mathbf{Q}^\text{teacher}_\text{sg}}{\tau_\text{cl}}\right), \qquad \boldsymbol{\pi}^\text{student} = \text{softmax}\!\left(\frac{\mathbf{Q}^\text{student}}{\tau_\text{cl}}\right), \qquad \tau_\text{cl} = 0.1$$

$$\mathcal{L}^\text{CL} = \text{KL}\!\left(\boldsymbol{\pi}^\text{teacher} \;\|\; \boldsymbol{\pi}^\text{student}\right) = \sum_a \pi^\text{teacher}_a \left(\log \pi^\text{teacher}_a - \log \pi^\text{student}_a\right)$$

Applied only on the memory portion of the replay buffer ($20\%$ of the batch).

**Weight**: $w_\text{cl} = 0.05$  
**Trains**: Encoder + Q-head  
**Optimizer**: main\_optimizer

### Total Loss (per Backward Pass)

$$\mathcal{L}^\text{main} = \mathcal{L}^\text{TD} + w_\text{cl}\, \mathcal{L}^\text{CL} \quad \text{(if CL enabled)}$$

$$\mathcal{L}^\text{lgdd} = \mathcal{L}^\text{InfoNCE} \quad \text{(separate backward pass)}$$

---

## 10. Continual Learning Extension

**File**: `src/learners/casvd_learner.py`, lines 441–485

When `cl_enabled=True`, CASVD supports continual learning scenarios where the task distribution changes over time (e.g., different map compositions).

### Teacher Network

An EMA copy of the main network serves as the teacher, initialised as $\theta^\text{teacher}_0 = \theta_0$, then updated after each training step:

$$\theta^\text{teacher}_j \leftarrow (1 - \tau_\text{cl})\, \theta^\text{teacher}_j + \tau_\text{cl}\, \theta_j, \qquad \tau_\text{cl} = 0.002$$

The teacher adapts slowly, providing a stable reference distribution.

### Distillation on Memory Buffer

The replay buffer is split:
- **Current batch** ($80\%$, ratio $\rho = 0.8$): recent experiences, trained with $\mathcal{L}^\text{TD}$ only
- **Memory batch** ($20\%$): older experiences, additionally trained with distillation

For memory samples $\mathcal{D}_\text{mem}$:

$$\mathbf{Q}^\text{student} = f_\theta(\mathbf{o}_\text{mem}), \qquad \mathbf{Q}^\text{teacher} = f_{\theta^\text{teacher}}(\mathbf{o}_\text{mem})\big|_\text{sg}$$

$$\boldsymbol{\pi}^\text{student} = \text{softmax}\!\left(\mathbf{Q}^\text{student} / 0.1\right), \quad \boldsymbol{\pi}^\text{teacher} = \text{softmax}\!\left(\mathbf{Q}^\text{teacher} / 0.1\right)$$

$$\mathcal{L}^\text{CL} = \mathbb{E}_{\mathcal{D}_\text{mem}}\!\left[\text{KL}\!\left(\boldsymbol{\pi}^\text{teacher} \| \boldsymbol{\pi}^\text{student}\right)\right]$$

This prevents catastrophic forgetting by ensuring the student policy does not drift too far from previously learned behaviours.

### Reservoir Buffer

A separate $\text{ReservoirReplayBuffer}$ (reservoir sampling) maintains a uniform sample from all past episodes, ensuring old experiences are not completely overwritten. Capacity $= 10{,}000$ episodes.

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

**Decision**: InfoNCE never touches the encoder. Embeddings are $\text{stop\_gradient}$-ed.

**Rationale**: If InfoNCE gradients flowed into the encoder, they would compete with Q-learning gradients. The encoder would be torn between making Q-values accurate ($\mathcal{L}^\text{TD}$) and making embeddings predictive of global future ($\mathcal{L}^\text{InfoNCE}$). These objectives can conflict — an agent might learn embeddings that are great for predicting the future but terrible for estimating Q-values. By detaching, InfoNCE acts as a read-only sensor: it observes embeddings to measure coordination but never pushes the encoder in any direction.

### 2. Q-Spread-Relative Alpha (Not Fixed Temperature)

**Decision**: Temperature $\alpha_i = \phi_i \cdot \Delta Q_i$ instead of a fixed constant.

**Rationale**: A fixed temperature would need careful tuning per environment and would become invalid as Q-values evolve during training. The Q-spread-relative formulation:
- Automatically adapts to any Q-value scale
- Is invariant to uniform Q-value scaling (mathematically proven in Section 3.7)
- Creates a self-stabilising feedback loop (Section 3.6)
- Works across different maps, agent counts, and training stages without retuning

### 3. Double-Q for Soft Values

**Decision**: Online network selects soft policy $\pi^\text{soft}$, target network evaluates Q-values.

**Rationale**: Standard soft values using a single network would overestimate because $\mathbb{E}_{a \sim \pi^\text{soft}(Q)}\!\left[Q(a)\right]$ uses the same Q-values to both create and evaluate the policy, leading to bias. Double-Q breaks this coupling: the online Q-values determine which actions get high probability, but the target Q-values (independently estimated) determine what those actions are actually worth.

### 4. Tanh Squashing with $\varsigma = 2.0$

**Decision**: $Q = \varsigma \cdot \tanh(Q^\text{raw})$ bounding Q-values to $[-\varsigma, +\varsigma] = [-2, +2]$.

**Rationale**: Unbounded Q-values can cause numerical instability in softmax computation (overflow/underflow), runaway bootstrapping, and difficulty for the mixer's hypernetworks. The $[-2, +2]$ range provides sufficient dynamic range while preventing these issues. The small $g_q = 0.1$ initialisation ensures Q-values start near zero, giving near-uniform initial soft policies.

### 5. Small Q-Head Initialisation ($g_q = 0.1$)

**Decision**: Q-head weights initialised with orthogonal gain $g_q = 0.1$ while encoder uses $g_\text{enc} = 1.0$.

**Rationale**: With $Q = 2\tanh(\mathbf{W}_q \mathbf{h})$: gain $0.1$ means initial $\|\mathbf{W}_q \mathbf{h}\|$ is small $\Rightarrow$ $\tanh(\text{small}) \approx \text{small}$ $\Rightarrow$ $Q \approx 0$ $\Rightarrow$ near-uniform softmax $\Rightarrow$ natural exploration without relying solely on $\varepsilon$-greedy. This prevents early training instability where random large Q-values could cause the soft policy to commit to bad actions before any learning has occurred.

### 6. Dual Optimisers with Different Learning Rates

**Decision**: Main network $\eta_\text{main} = 10^{-3}$, InfoNCE predictor $\eta_\text{lgdd} = 3 \times 10^{-4}$.

**Rationale**: The InfoNCE predictor is a simple linear layer that converges quickly. If trained at the same rate as the main network, it would overfit to current embeddings and oscillate. The lower learning rate and complete gradient isolation ensure the coordination sensor provides stable, slowly-evolving signals that do not disrupt Q-learning.

### 7. Strong EMA Smoothing for Coordination Signals ($\tau_\text{ema} = 0.999$)

**Decision**: Coordination signals smoothed with EMA $\tau_\text{ema} = 0.999$ (very slow update).

**Rationale**: The InfoNCE loss is noisy (depends on random negative sampling and batch composition). Without strong smoothing, the per-agent $\phi_i$ values would oscillate wildly, destabilising soft value computation. The $\tau_\text{ema} = 0.999$ means the signal changes very gradually — approximately $1000$ updates to reach a new equilibrium — providing a stable coordination measure.

### 8. Entity Masking for Dead/Invisible Entities

**Decision**: Zero-feature entities are masked to $-\infty$ before attention softmax.

**Rationale**: In SMACv2, dead or out-of-sight entities have all-zero features. Without masking, attention would still distribute weight to these ghost entities, polluting the agent's local summary $\boldsymbol{\ell}^\text{loc}_{b,i}$ with meaningless information. The $-\infty$ mask ensures zero attention weight after softmax, effectively removing dead entities from the attention computation.

---

## 13. File Reference Map

| Component | File Path | Key Lines | Key Classes/Functions |
|-----------|-----------|-----------|----------------------|
| **Main Entry** | `src/main.py` | 27–38 | `my_main()` |
| **Training Loop** | `src/run/run.py` | 76–273 | `run_sequential()` |
| **CASVD Learner** | `src/learners/casvd_learner.py` | 35–485 | `CASVDLearner` |
| **CASVD Controller** | `src/controllers/casvd_controller.py` | 6–116 | `CASVDMAC` |
| **GATNSAgent** | `src/modules/agents/gat_ns_agent.py` | 118–305 | `GATNSAgent`, `LocalEntityGAT`, `TeamGATLayer` |
| **LocalEntityGAT** | `src/modules/agents/gat_ns_agent.py` | 11–43 | `LocalEntityGAT` |
| **TeamGATLayer** | `src/modules/agents/gat_ns_agent.py` | 46–71 | `TeamGATLayer` |
| **InfoNCE Predictor** | `src/modules/predictors/infonce_predictor.py` | 24–80 | `InfoNCEPredictor` |
| **NMixer** | `src/modules/mixers/nmix.py` | 8–71 | `NMixer` |
| **QMIX Mixer** | `src/modules/mixers/qmix.py` | 7–64 | `QMixer` |
| **Episode Buffer** | `src/components/episode_buffer.py` | — | `ReplayBuffer`, `EpisodeBatch` |
| **Action Selectors** | `src/components/action_selectors.py` | — | `EpsilonGreedyActionSelector` |
| **Episode Runner** | `src/runners/episode_runner.py` | 56–125 | `EpisodeRunner.run()` |
| **Parallel Runner** | `src/runners/parallel_runner.py` | 14–120 | `ParallelRunner` |
| **SMACv2 Wrapper** | `src/envs/smacv2_wrapper.py` | 38–187 | `SMACv2Wrapper` |
| **CASVD Config** | `src/config/algs/casvd.yaml` | 1–108 | YAML configuration |
| **Default Config** | `src/config/default.yaml` | — | Base hyperparameters |
| **Env Config** | `src/config/envs/sc2v2.yaml` | — | SMACv2 environment settings |
| **Map Configs** | `src/config/envs/smacv2_configs/*.yaml` | — | Per-map scenario configs |
