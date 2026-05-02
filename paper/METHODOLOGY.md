# Methodology — GAT + Soft-QMIX + Components 1+2+3

A complete mathematical and architectural specification of the proposed method.

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| $n$ | number of agents (5 in SMACv2 Protoss 5v5) |
| $\mathcal{A}$ | discrete action set per agent ($|\mathcal{A}| = $ 12 in SMACv2) |
| $\mathcal{S}$ | global state space |
| $o_i^t \in \mathcal{O}_i$ | local observation of agent $i$ at time $t$ |
| $\tau_i^t = (o_i^0, a_i^0, \dots, o_i^t)$ | observation–action history of agent $i$ |
| $r_t \in \mathbb{R}$ | shared team reward at step $t$ |
| $\gamma \in [0,1)$ | discount factor (0.99) |
| $\lambda \in [0,1]$ | TD($\lambda$) parameter (0.4) |
| $\pi_i(a \mid \tau_i)$ | per-agent stochastic policy |
| $Q_i(\tau_i, a)$ | per-agent action value |
| $Q_{\text{tot}}(s, \mathbf{a})$ | joint action value, mixed from $\{Q_i\}$ |
| $\alpha_i \in \mathbb{R}_+$ | per-agent entropy coefficient (sampling temperature) |
| $\bar\alpha = \frac{1}{n}\sum_i \alpha_i$ | mean entropy coefficient (used in target — Component 1) |
| $h_i^t \in \mathbb{R}^d$ | per-agent hidden state from the encoder |
| $\ell_i^t \in \mathbb{R}^d$ | per-agent local summary (pre-TeamGAT) |
| $f_i, g_i$ | learned mixer operators per agent |
| $\rho(t) \in [0,1]$ | curriculum ramp at training step $t$ (Component 3) |

We work in a Dec-POMDP $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}^n, P, R, \mathcal{O}_1, \dots, \mathcal{O}_n, n, \gamma \rangle$ with a single shared reward signal. Centralised training, decentralised execution.

---

## 1. Architecture Overview

The proposed method GAT + Soft-QMIX + Components 1+2+3 (henceforth **CASVD-123**) is the composition of three architectural pieces and three algorithmic components:

```
┌──────────────────────┐    ┌──────────────────────┐    ┌─────────────────┐
│  GAT Encoder         │    │   Soft-QMIX Backbone │    │  Components     │
│  (per-agent → h_i)   │ →  │  (Q_i → Q_tot)       │ →  │  1, 2, 3        │
│                      │    │                      │    │                 │
│  • LocalEntityGAT    │    │  • VDN sum mixer     │    │  C1: decoupled  │
│  • TeamGAT (cross-   │    │  • func_g shaping    │    │      α          │
│    agent attention)  │    │  • func_f temp       │    │  C2: Q-spread   │
│                      │    │  • TD(λ) targets     │    │      sensor     │
│  • Q-head            │    │  • β-loss + W-TD     │    │  C3: curriculum │
└──────────────────────┘    └──────────────────────┘    └─────────────────┘
```

The encoder produces per-agent Q-values; the Soft-QMIX backbone defines the policy and target; the three components stabilise training by (1) decoupling the sampling-side and target-side α, (2) deriving per-agent α from a Q-spread sensor, and (3) phasing in the heterogeneity over a curriculum window.

---

## 2. The GAT Encoder

The encoder is a two-stage graph attention block that maps per-agent observations to per-agent hidden states.

### 2.1 Per-agent input

Following SMACv2 convention, the per-agent input at step $t$ is
$$
x_i^t \;=\; \big[\, o_i^t \,\Vert\, a_i^{t-1} \,\Vert\, \mathrm{onehot}(i) \,\big] \;\in\; \mathbb{R}^{d_o + |\mathcal{A}| + n}.
$$
$o_i^t$ is the local observation, $a_i^{t-1}$ the previous action (one-hot), and the agent identity one-hot. We initialise $a_i^{-1} = \mathbf{0}$.

### 2.2 Stage 1 — Local entity attention

Each agent's observation is decomposed into entities (own unit, visible enemies, visible allies). Let $E_i^t = \{e_{i,1}, \dots, e_{i,m_i}\}$ be the entity set; each $e_{i,j} \in \mathbb{R}^{d_e}$. Multi-head attention with $H$ heads:

$$
q_i = W_q^{(h)} e_{i,\text{self}}, \qquad
k_{i,j} = W_k^{(h)} e_{i,j}, \qquad
v_{i,j} = W_v^{(h)} e_{i,j},
$$

$$
\alpha_{i,j}^{(h)} = \frac{\exp(q_i^\top k_{i,j} / \sqrt{d_e/H})}{\sum_{j'} \exp(q_i^\top k_{i,j'} / \sqrt{d_e/H})}.
$$

The local summary is
$$
\ell_i^t \;=\; \mathrm{LN}\!\left( e_{i,\text{self}} + \sum_{h=1}^{H} W_o^{(h)} \!\sum_{j=1}^{m_i} \alpha_{i,j}^{(h)} \, v_{i,j}^{(h)} \right) \;\in\; \mathbb{R}^d.
$$

LayerNorm + residual is applied; we use $H = 4$ heads, $d = 128$.

### 2.3 Stage 2 — Cross-agent attention (TeamGAT)

Each agent attends over all teammates' local summaries. Define the teammate set as $\{\ell_j^t : j \neq i\}$; agent $i$'s self-attention with teammate keys/values:

$$
h_i^t \;=\; \mathrm{LN}\!\left( \ell_i^t + \sum_{h=1}^{H} W_{\text{out}}^{(h)} \!\sum_{j \neq i} \beta_{i,j}^{(h)} \, V_h \ell_j^t \right),
$$

where $\beta_{i,j}^{(h)} = \mathrm{softmax}_j (Q_h \ell_i^t)^\top (K_h \ell_j^t) / \sqrt{d/H}$. We mask dead teammates with `alive_mask`. The output dimension is preserved at $d = 128$.

### 2.4 Q-head

A single linear layer maps $h_i^t$ to action values:
$$
Q_i(\tau_i^t, \cdot) \;=\; W_Q\, h_i^t + b_Q \;\in\; \mathbb{R}^{|\mathcal{A}|}.
$$
$W_Q$ is initialised with orthogonal init at gain $0.1$ to keep early-training Q values modest.

The encoder outputs per-step are $\{Q_i\}_{i=1}^n$ and the auxiliary latents $\{\ell_i, h_i\}$ used for diagnostics.

---

## 3. Soft-QMIX Backbone

Following Soft-QMIX, the joint action–value is composed via **VDN-style sum** with two learned per-agent operators.

### 3.1 The mixer

$$
Q_{\text{tot}}(s, \mathbf{a}) \;=\; \sum_{i=1}^{n} Q_i(\tau_i, a_i),
$$
i.e. pure VDN sum. The two learned operators act per-agent:

**$g(\cdot)$ — order-preserving residual** (shapes the Q landscape per agent without changing action ranking):
$$
g_i(Q_i)(a) \;=\; Q_i(a) \;+\; r_i(s) \cdot \big(Q_i(a) - \mathrm{softmax}_a(Q_i)\big)_a,
$$
where $r_i(s) \geq 0$ is a state-conditioned residual gain produced by a hypernetwork.

**$f(\cdot)$ — affine per-agent temperature**:
$$
f_i(Q_i)(a) \;=\; w_i(s) \cdot Q_i(a) \,+\, b_i(s),
$$
with $w_i, b_i$ from a hypernetwork. $w_i \geq 0$ is enforced.

### 3.2 Soft policy

Action selection during training uses

$$
\boxed{\;\pi_i^{\alpha}(a \mid s) \;=\; \mathrm{softmax}_a\!\left( \frac{f_i(g_i(Q_i(s, \cdot)))}{\alpha_i} \right)\;}
$$

with masking on unavailable actions (logit set to $-\infty$). At test time we use greedy $\arg\max_a Q_i$ on raw $Q$ (no $f, g$, no temperature).

### 3.3 TD($\lambda$) target with entropy bonus

Sample-based entropy estimate for a single roll-out trajectory; target action $a^*_{t+1}$ is sampled from the **online** policy under the current $\alpha$ (Double-Q):

$$
a^*_{t+1} \,\sim\, \pi^{\alpha}(\cdot \mid s_{t+1}; \theta_{\text{online}}),
$$

$$
y_t \;=\; r_t \,+\, \gamma \!\left( \underbrace{Q_{\text{tot}}^{\text{tgt}}(s_{t+1}, a^*_{t+1})}_{\text{target net evaluates}} \;+\; \underbrace{\bar\alpha \sum_{i=1}^{n} \mathcal{H}\big(\pi_i^{\alpha}(\cdot \mid s_{t+1})\big)}_{\text{entropy bonus — Component 1: scalar }\bar\alpha} \right).
$$

Practically we use the *single-sample* entropy estimator $\mathcal{H}(\pi_i) \approx -\log \pi_i(a^*_i)$, so

$$
y_t \;=\; r_t + \gamma \!\left( Q_{\text{tot}}^{\text{tgt}}(s_{t+1}, a^*_{t+1}) \;-\; \bar\alpha \sum_{i=1}^{n} \log \pi_i^{\alpha}(a^*_{t+1, i} \mid s_{t+1}) \right).
$$

The TD($\lambda$) target chains these one-step targets in the standard backward-view manner with bootstrapping mask:

$$
y_t^{(\lambda)} \;=\; r_t + (1-\lambda)\gamma\, Q^{\text{tgt}}_{t+1} \;+\; \lambda\gamma\big(y_{t+1}^{(\lambda)} + \alpha\text{-bonus}_{t+1}\big),
$$

with the boundary $y_T^{(\lambda)} = Q^{\text{tgt}}_T \cdot (1 - \mathrm{terminated}_T)$.

### 3.4 Three loss terms

**TD loss** (for monitoring; not used directly):
$$
\mathcal{L}_{\text{TD}} = \tfrac{1}{2}\,\mathbb{E}_{(s,\mathbf{a},r,s')}\!\left[\big(Q_{\text{tot}}(s, \mathbf{a}) - y_t^{(\lambda)}\big)^2\right].
$$

**Beta loss** keeps the affine $f$ near identity by penalising the gap between mixed Q and the agent-summed affine Q:
$$
\delta_t \;=\; Q_{\text{tot}}(s_t, \mathbf{a}_t) \;-\; \sum_i f_i(g_i(Q_i))(a_{i,t}),
$$
$$
\mathcal{L}_{\beta} \;=\; \tfrac{1}{2}\,\mathbb{E}\big[\delta_t^2\big].
$$

**Weighted-TD loss** (the actual training loss for the value head) — uses an asymmetric mask that down-weights the sample when the over/under-shoot directions of $\delta_t$ and TD-error agree:
$$
m_{\text{gopt}} \;=\; \mathbb{1}\big[(\delta_t > 0) \,\not\Leftrightarrow\, (\text{td\_err}_t < 0)\big],
$$
$$
\mathcal{L}_{\text{WTD}} \;=\; \frac{\sum_t \big(\tfrac{1}{2}\,\text{td\_err}_t^2 \cdot \tfrac{1}{2}\, m_{\text{gopt}, t} + \tfrac{1}{2}\,\text{td\_err}_t^2 \cdot (1 - m_{\text{gopt}, t})\big)}{\sum_t \big(\tfrac{1}{2}\, m_{\text{gopt}, t} + (1 - m_{\text{gopt}, t})\big)}.
$$

The **total Soft-QMIX loss** is
$$
\boxed{\;\mathcal{L} \;=\; \mathcal{L}_{\text{WTD}} \;+\; \mathcal{L}_{\beta}.\;}
$$

---

## 4. The Failure Mode We Are Fixing

Before specifying the components, formalise the failure mode they remedy.

### 4.1 The cumulative entropy bonus

For a $T$-step trajectory under policy $\pi$, the discounted reward + entropy objective evaluates to

$$
J(\pi) \;=\; \mathbb{E}_\pi\!\left[\sum_{t=0}^{T-1} \gamma^t \!\left(r_t + \alpha \sum_i \mathcal{H}\big(\pi_i(\cdot \mid s_t)\big)\right)\right].
$$

If the per-step team entropy averages $\bar H = \tfrac{1}{T}\sum_t \sum_i \mathcal{H}(\pi_i(\cdot \mid s_t))$, the *contribution* of the entropy term is

$$
B(T, \alpha) \;\approx\; \alpha \cdot T \cdot \bar H \cdot \frac{1 - \gamma^T}{1 - \gamma}.
$$

$B$ scales monotonically with $T$ for $\gamma < 1$. **A policy that produces longer episodes harvests more entropy bonus, holding $\bar H$ fixed.**

### 4.2 Asymmetric per-agent $\alpha$ — the stalling attractor

Suppose the per-agent $\alpha_i$ are heterogeneous. The objective becomes

$$
J(\pi) \;=\; \mathbb{E}_\pi\!\left[\sum_t \gamma^t \!\left(r_t + \sum_i \alpha_i \,\mathcal{H}(\pi_i(\cdot \mid s_t))\right)\right].
$$

Take the policy gradient w.r.t. parameters $\theta$. The entropy term contributes

$$
\nabla_\theta \!\left[\alpha_i \,\mathcal{H}(\pi_i)\right] \;=\; \alpha_i \cdot \nabla_\theta \mathcal{H}(\pi_i).
$$

Agents with larger $\alpha_i$ get a *stronger* gradient pulling toward higher-entropy (more uniform) policies. In a cooperative team where the joint policy must be coordinated, the team's only consistent solution is to pick a joint action *that does not depend on the high-$\alpha$ agent committing*. The simplest such joint policy is **stalling** — staying alive without committing to coordinated offensive maneuvers, which absorbs many timesteps while the team waits.

We empirically confirm this in our prior Run B (per-agent $\alpha = [0.01, 0.02, 0.03, 0.04, 0.05]$, hardcoded): episode length grows from 95 to 134, dead-allies-per-episode falls from 3.60 to 2.99, win rate collapses from 0.695 (peak) to 0.512 (last-50 mean) — the same signature as Run A but in extreme form.

### 4.3 Encoder-induced asymmetric $\alpha$

The same failure mode arises *implicitly* even when nominal $\alpha$ is shared. With the GAT encoder, the per-agent Q-vectors $Q_i(\tau_i, \cdot)$ exhibit different *spreads* across agents. Since the policy is $\pi_i(a) \propto \exp(f_i(g_i(Q_i))(a) / \alpha)$, the *effective* exploration temperature depends on the spread:

$$
\alpha_i^{\text{eff}} \;\propto\; \frac{\alpha}{\mathrm{spread}(Q_i)}.
$$

If the encoder produces per-agent Q-vectors with non-uniform spreads, the effective per-agent $\alpha^{\text{eff}}$ varies, reproducing §4.2's asymmetric pressure even with a shared nominal $\alpha$.

This is why Run A (GAT + scalar $\alpha = 0.03$) exhibits late-training stalling: the encoder makes the regularisation silently per-agent, and the optimiser slides into the stalling attractor.

---

## 5. Component 1 — Decoupled Sampling-α from Target-α

**Idea**: $\alpha_i$ appears in two structurally different places. Use heterogeneous $\alpha_i$ in *sampling* (where heterogeneity may help exploration), but use the uniform mean $\bar\alpha$ in the *target* (where heterogeneity causes stalling).

### 5.1 Specification

Let $\bar\alpha = \tfrac{1}{n}\sum_i \alpha_i$ (or a hyperparameter, identical to the original Soft-QMIX scalar coefficient). The training is modified at exactly two equations:

**Sampling** uses per-agent $\alpha_i$:
$$
\pi_i^{\alpha_i}(a \mid s) \;=\; \mathrm{softmax}_a\!\left(\frac{f_i(g_i(Q_i))(a)}{\alpha_i}\right).
$$

**Target** uses uniform $\bar\alpha$:
$$
\boxed{\;y_t \;=\; r_t + \gamma\!\left( Q_{\text{tot}}^{\text{tgt}}(s_{t+1}, a^*_{t+1}) \;-\; \bar\alpha \sum_i \log \pi_i^{\alpha_i}(a^*_{t+1, i} \mid s_{t+1}) \right).\;}
$$

Note carefully: the policy $\pi_i^{\alpha_i}$ in the bonus is still computed with the per-agent $\alpha_i$. Only the *coefficient* on $\sum_i \log \pi_i$ is uniform.

### 5.2 Why this prevents the stalling attractor

The §4.2 gradient becomes

$$
\nabla_\theta\!\left[\bar\alpha \sum_i \mathcal{H}(\pi_i^{\alpha_i})\right] \;=\; \bar\alpha \sum_i \nabla_\theta \mathcal{H}(\pi_i^{\alpha_i}).
$$

The pull toward high-entropy joint policies is now uniform across agents (weighted by $\bar\alpha$, not by $\alpha_i$). The *direction* of the gradient is the same as scalar-$\alpha$ Soft-QMIX. The asymmetric pressure that produced stalling is removed.

The exploration *distribution* still benefits from per-agent $\alpha_i$ in the sampling step — agents with high $\alpha_i$ produce broader policies, agents with low $\alpha_i$ commit, exactly as intended.

### 5.3 Reduction to scalar Soft-QMIX

When $\alpha_i = \bar\alpha$ for all $i$, Component 1 reduces exactly to standard Soft-QMIX. Component 1 is a **strict generalisation** that costs nothing if heterogeneity is not exploited.

---

## 6. Component 2 — Q-spread Per-Agent Sensor

**Idea**: drive per-agent $\alpha_i$ from a state-conditioned signal of agent confidence, computed cheaply from quantities the learner already produces.

### 6.1 The signal

For each batch and each agent $i$, define the per-step Q-spread

$$
s_i(t) \;=\; \max_{a \,\in\, \mathrm{avail}(i, t)} Q_i(\tau_i^t, a) \;-\; \min_{a \,\in\, \mathrm{avail}(i, t)} Q_i(\tau_i^t, a).
$$

Mask unavailable actions (replace by $\pm\infty$ for max/min) and exclude per-(batch, time, agent) entries where the agent has no available actions. Aggregate over a batch with a valid-step mask $m_{b,t,i}$:

$$
\bar s_i^{\text{batch}} \;=\; \frac{\sum_{b,t} m_{b,t,i} \cdot s_i(b, t)}{\sum_{b,t} m_{b,t,i}}.
$$

Maintain a per-agent EMA with time-constant $\tau_s = 0.99$:

$$
\hat s_i^{(k)} \;=\; \tau_s\, \hat s_i^{(k-1)} \;+\; (1 - \tau_s)\, \bar s_i^{\text{batch}, (k)},
$$

where $k$ indexes the train step.

### 6.2 Confidence and α mapping

Per-team-mean normalised confidence:
$$
c_i \;=\; \mathrm{clamp}\!\left(\frac{\hat s_i}{\frac{1}{n}\sum_j \hat s_j},\; c_{\min},\; c_{\max}\right), \qquad c_{\min} = 0.3,\; c_{\max} = 3.0.
$$

The clamp prevents degenerate agents (greedy near-collapse or near-uniform) when the spread ratios are extreme.

Inverse-confidence map to $\alpha$:

$$
\boxed{\;\alpha_i \;=\; \frac{\bar\alpha}{c_i} \;\in\; \left[\frac{\bar\alpha}{c_{\max}},\; \frac{\bar\alpha}{c_{\min}}\right] \;=\; [0.01,\, 0.10] \text{ for } \bar\alpha = 0.03.\;}
$$

**Interpretation**: an agent with wide Q-spread (clearly best action; high $c_i$) gets a *low* $\alpha_i$ and commits. An agent with narrow Q-spread (uncertain; low $c_i$) gets a *high* $\alpha_i$ and hedges.

### 6.3 Computational cost

Q-spread is computed from `mac_out` which the learner produces on every train step. The EMA is a length-$n$ tensor update. **No additional networks, no auxiliary loss, no second optimiser.** The added per-step cost is $O(BTn|\mathcal{A}|)$ for the max/min reduction — negligible relative to the main forward/backward pass.

### 6.4 Why Q-spread instead of InfoNCE on hidden states?

Earlier work (and our own previous attempts) drove per-agent $\alpha$ from contrastive prediction over encoder hidden states. In role-randomised SMACv2, this fails: the per-agent identity that the contrastive signal would need to detect *does not exist as a stable feature* of the encoder representation. Empirically, our InfoNCE-based `coord_signal_std` stayed at ~0.003 across 10M training steps, even when the *underlying* per-agent $\alpha$ was explicitly heterogeneous (Run B).

Q-spread bypasses this entirely. It does not require any *identity* feature; it requires only that the *current state* induces different decision confidence in different agents — which is a state-conditioned property that *does* exist in role-randomised tasks (e.g. an agent currently in close combat has a clearer best action than one in open positioning).

Empirically, Q-spread `q_spread_std` reaches ~0.012 — **3.4× more discriminative** than the InfoNCE signal.

---

## 7. Component 3 — Curriculum from Uniform to Heterogeneous

**Idea**: heterogeneous $\alpha$ is only safe to introduce *after* the team has learned a basic coordinated joint policy under symmetric conditions.

### 7.1 The ramp

Define a curriculum ramp over training step $t$ (in environment steps):

$$
\rho(t) \;=\;
\begin{cases}
0 & t \leq t_0 \\
\dfrac{t - t_0}{t_1 - t_0} & t_0 < t < t_1 \\
1 & t \geq t_1
\end{cases}
$$

with default $t_0 = 2 \times 10^6,\; t_1 = 4 \times 10^6$.

### 7.2 Effective α

The actual $\alpha$ used in *both* sampling and target is

$$
\boxed{\;\alpha_i^{\mathrm{eff}}(t) \;=\; \bar\alpha \;+\; \rho(t) \cdot (\alpha_i(t) - \bar\alpha),\;}
$$

where $\alpha_i(t)$ is the heterogeneous value from Component 2 at step $t$.

### 7.3 Behavioural phases

- **Phase 1** ($t \leq t_0 = 2$M): $\alpha_i^{\mathrm{eff}} = \bar\alpha$ for all $i$. The system runs as **vanilla scalar Soft-QMIX**. The team learns the basic coordinated policy.
- **Phase 2** ($t \in [t_0, t_1]$): heterogeneity fades in linearly. The team adapts to gradually increasing per-agent asymmetry.
- **Phase 3** ($t \geq t_1 = 4$M): $\alpha_i^{\mathrm{eff}} = \alpha_i$. Full Q-spread-driven heterogeneity is in effect.

### 7.4 Why a curriculum is necessary

In the controlled premise test (our Run B, per-agent $\alpha$ from step zero), the team never recovered from the early disruption: heterogeneity from $t=0$ prevented the formation of a coordinated joint policy. The curriculum ensures that by the time heterogeneity is introduced, the team already has a *symmetric* attractor that has captured the reward gradient strongly enough to dominate the §4.1 entropy pressure.

Empirically: in Run C (curriculum on), `alpha_std` is exactly 0 until $t = 2$M, climbs cleanly to ~0.003 by $t = 4$M, and stays there. The team's win rate is on Run A's trajectory through $t \leq t_0$ and only diverges *upward* in the [4M, 7M] window.

---

## 8. Combined Algorithm

The full per-train-step procedure:

```
INPUT: minibatch B = {(τ, a, r, τ', terminated, mask, state, avail)}
         with shapes [B, T, ...]; train step counter t_env

# ── Forward pass ───────────────────────────────────────────────
mac.init_hidden(batch_size)
for t in 0..T-1:
    Q_t  ← encoder(τ[:, t]; θ)             # GAT → per-agent Q [B, n, |A|]
    push Q_t into mac_out
mac_out  ← stack(mac_out)                   # [B, T, n, |A|]
mac_out  ← mixer.func_g(mac_out, state)     # order-preserving shaping

q_taken  ← gather(mac_out[:-1], a)          # [B, T-1, n]

# ── Component 2 — Q-spread sensor update (online net) ─────────
if alpha_mode == "q_spread_adaptive":
    s_i  ← max_a Q_i(s, a) − min_a Q_i(s, a)  on avail actions, masked
    update EMA  ŝ_i  with τ = 0.99
    c_i  ← clamp(ŝ_i / mean(ŝ), 0.3, 3.0)
    α_i  ← α_mean / c_i

# ── Component 3 — curriculum blend ─────────────────────────────
ρ     ← ramp(t_env)
α_eff ← α_mean + ρ * (α_i − α_mean)         # [n]

# ── Target computation (no grad) ───────────────────────────────
target_mac_out  ← target_encoder forward over T steps
target_mac_out  ← target_mixer.func_g(target_mac_out, state)

# Online policy for action sampling (Double-Q)
Q_for_pi  ← mixer.func_f(mac_out.detach(), state)
logits    ← Q_for_pi / α_eff[None, None, :, None]         # broadcast α per agent
mask unavail actions; π ← softmax(logits)

a_star    ← cdf-sample from π                              # [B, T, n]
Q_target_at_a_star  ← gather(target_mac_out, a_star)
log π(a_star)       ← gather(log(π+ε), a_star)             # [B, T, n]

# ── Component 1 — scalar α_mean in target entropy bonus ────────
target_entropy  ← − α_mean · sum_i log π(a_star)_i         # [B, T, 1]

# Mix sampled target Q through VDN sum
Q_tot_target  ← mixer.target.sum_i Q_target_at_a_star_i

# TD(λ) backward recursion
y  ← build_td_lambda(r, terminated, mask, Q_tot_target, target_entropy, γ, λ)

# ── Loss computation ───────────────────────────────────────────
Q_tot          ← mixer(q_taken, state[:-1])
td_err         ← Q_tot − y.detach()
δ              ← Q_tot.detach() − sum_i mixer.func_f(Q_i)(a_i)
m_gopt         ← ¬((δ > 0) ⊕ (td_err < 0))
L_TD           ← weighted-mean(0.5 * td_err²)               # for logging
L_β            ← weighted-mean(0.5 * δ²)
L_WTD          ← (Σ td_err² × (m_gopt/2 + (1−m_gopt)) ) /
                  (Σ (m_gopt/2 + (1−m_gopt)))
L              ← L_WTD + L_β

# ── Optimise ───────────────────────────────────────────────────
optimizer.zero_grad(); L.backward();
clip_grad(params, 10); optimizer.step()

# Push α_eff to action selector for next rollout
mac.set_alpha(α_eff.detach())

# Target net updates (every 200 train episodes, hard copy)
if episode_num − last_update ≥ 200:
    target_encoder.load_state(encoder)
    target_mixer.load_state_dict(mixer.state_dict())
```

---

## 9. Hyperparameters

| Hyperparameter | Symbol | Value |
|---|---|---|
| Discount factor | $\gamma$ | 0.99 |
| TD($\lambda$) coefficient | $\lambda$ | 0.4 |
| Mean entropy coefficient | $\bar\alpha$ | 0.03 |
| Q-spread EMA time-constant | $\tau_s$ | 0.99 |
| Confidence clamp lower | $c_{\min}$ | 0.3 |
| Confidence clamp upper | $c_{\max}$ | 3.0 |
| Curriculum start | $t_0$ | $2 \times 10^6$ |
| Curriculum end | $t_1$ | $4 \times 10^6$ |
| GAT hidden dim | $d$ | 128 |
| GAT attention heads | $H$ | 4 |
| Q-head orthogonal gain | — | 0.1 |
| Encoder orthogonal gain | — | 1.0 |
| Replay buffer size | — | 5000 episodes |
| Batch size | — | 128 episodes |
| Parallel envs | — | 8 |
| Target update interval | — | 200 episodes (hard copy) |
| Optimiser | — | Adam |
| Learning rate | — | $10^{-3}$ |
| Adam $\epsilon$ | — | $10^{-7}$ |
| Gradient clip | — | $\|\nabla\| \leq 10$ |

---

## 10. Information Flow Summary

```
       ┌────────────┐
o, a-1 │            │              ┌─────────────┐
   ──→ │  GAT       │ ── Q_i ───→  │             │
       │  Encoder   │              │  Soft-QMIX  │ ── Q_tot ─┐
       │            │              │  (g, f, ⊕)  │           │
       │ ── ℓ_i ──→ (diagnostic)   │             │           │
       │ ── h_i ──→ (diagnostic)   └─────────────┘           │
       └────────────┘                     │                  │
                                          │                  │
                                          ↓                  │
                                  ┌──────────────┐           │
                                  │  Q-spread    │ ─ ŝ_i ──→ Component 2:
                                  │  sensor      │            α_i = α_mean / clamp(ŝ_i/⟨ŝ⟩)
                                  │  (Comp 2)    │
                                  └──────────────┘                  │
                                                                    │
                                  ┌──────────────┐                  │
                                  │  curriculum  │ ─ ρ(t) ──→  α_eff = α_mean + ρ(α_i − α_mean)
                                  │  ramp ρ(t)   │ ←──┐
                                  │  (Comp 3)    │    t_env
                                  └──────────────┘                  │
                                                                    │
                                  ┌──────────────────────────┐      │
                                  │  Sampling: π_i = sm(Q/α_eff_i)│ ←┤
                                  │                          │      │
                                  │  Target  : y = r + γ(Q'  │      │
                                  │            − α_mean · Σ_i log π) │ ← Component 1
                                  └──────────────────────────┘     (scalar α_mean)
```

---

## 11. Computational Complexity

Let $T$ be episode length, $B$ batch size, $n$ agents, $|\mathcal{A}|$ action set, $d$ encoder dim, $H$ heads.

- GAT encoder forward: $O(BTn(d^2 + n d))$ — local entity attention + cross-agent.
- Soft-QMIX mixer: $O(BTn d)$ for $f, g$ (linear in agents).
- Sample-based entropy: $O(BTn|\mathcal{A}|)$.
- Q-spread sensor update: $O(BTn|\mathcal{A}|)$.
- Curriculum ramp: $O(1)$.

Total per train step is dominated by the encoder. **Components 2 and 3 add $O(n)$ extra parameters to track (the EMA + ramp scalar) and no measurable wall-clock overhead**.

---

## 12. Reduction Identities

For the reader sceptical that Components 1+2+3 are not just hidden hyperparameters:

1. **If $\alpha_i = \bar\alpha$ for all $i$**: Components 1, 2 reduce to scalar-$\alpha$ Soft-QMIX. Component 3 is a no-op.
2. **If $t_0 = t_1 = 0$ (no curriculum)**: Component 3 is a no-op.
3. **If $c_{\min} = c_{\max} = 1$ (collapsed clamp)**: Component 2 forces $\alpha_i = \bar\alpha$, reducing to scalar.
4. **If $\bar\alpha$ in the target is replaced by per-agent $\alpha_i$**: we recover the failure case (Run B). Component 1 is exactly the mathematical operation that prevents this.

These reductions establish the components as a strict, principled extension of Soft-QMIX with one knob each (heterogeneity range, curriculum window, target uniformity).

---

## 13. Summary in One Equation

The proposed CASVD-123 method is, *all of it*, the modification of Soft-QMIX defined by replacing the standard scalar-α target

$$
y_t^{\mathrm{Soft-QMIX}} \;=\; r_t + \gamma\!\left(Q_{\text{tot}}^{\text{tgt}} \,-\, \alpha \sum_i \log \pi_i^{\alpha}(a^*_i)\right)
$$

with the curriculum-blended, decoupled, sensor-driven target

$$
\boxed{\;y_t^{\mathrm{CASVD\text{-}123}} \;=\; r_t + \gamma\!\left(Q_{\text{tot}}^{\text{tgt}} \,-\, \bar\alpha \sum_i \log \pi_i^{\alpha_i^{\mathrm{eff}}(t)}(a^*_i)\right),\;}
$$

where

$$
\alpha_i^{\mathrm{eff}}(t) \;=\; \bar\alpha \,+\, \rho(t) \cdot \left(\frac{\bar\alpha}{c_i} - \bar\alpha\right), \qquad
c_i \;=\; \mathrm{clamp}\!\left(\frac{\hat s_i}{\langle \hat s \rangle},\; c_{\min},\; c_{\max}\right).
$$

This single equation encodes:

- The **decoupling** (Component 1): the coefficient on $\sum_i \log \pi_i$ is the scalar $\bar\alpha$, while the policy $\pi_i^{\alpha_i^{\mathrm{eff}}(t)}$ inside the log uses the per-agent value.
- The **sensor** (Component 2): $c_i$ is computed from Q-spread.
- The **curriculum** (Component 3): the ramp $\rho(t)$ governs how much heterogeneity is in effect at training step $t$.

When $\rho = 0$, $\alpha_i^{\mathrm{eff}} = \bar\alpha$, and the equation collapses to standard scalar Soft-QMIX. When $\rho = 1$, the full sensor-driven heterogeneity is in effect — but only in *sampling*, while the *coefficient* $\bar\alpha$ on the entropy bonus stays uniform, preventing the stalling attractor.

---

*End of methodology specification. See `paper.md` and `paper_final.pdf` for the full empirical study and results.*
