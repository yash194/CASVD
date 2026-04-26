"""Build the workshop paper PDF.  Uses fpdf2 for typesetting + PIL for image dims."""
from fpdf import FPDF
from PIL import Image
import os, re

HERE = os.path.dirname(os.path.abspath(__file__))


class Paper(FPDF):
    # Page geometry — single column, journal-ish margins
    L_MARGIN = 18
    R_MARGIN = 18
    T_MARGIN = 18
    B_MARGIN = 20

    # Use Unicode TTF (Arial) so we can include α, γ, σ, →, —, ≈, etc.
    F_REG  = ("Arial", "",  10.0)
    F_BOLD = ("Arial", "B", 10.0)
    F_ITAL = ("Arial", "I", 10.0)
    F_TITLE  = ("Arial", "B", 16.0)
    F_AUTHOR = ("Arial", "I", 11.0)
    F_H1 = ("Arial", "B", 13.0)
    F_H2 = ("Arial", "B", 11.5)
    F_H3 = ("Arial", "B", 10.5)
    F_CAPTION = ("Arial", "I", 9.0)
    F_TABLE = ("Arial", "", 9.0)
    F_TABLE_BOLD = ("Arial", "B", 9.0)

    def header(self):
        # No page header — cleaner workshop look
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"{self.page_no()}", align="C")
        self.set_text_color(0, 0, 0)

    # ---------- helpers ----------
    def W(self):
        return self.w - self.l_margin - self.r_margin

    def set_f(self, spec):
        self.set_font(*spec)

    def gap(self, h=2.5):
        self.ln(h)

    def hr(self, h=1.2):
        self.ln(h)
        self.set_draw_color(180, 180, 180)
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.set_draw_color(0, 0, 0)
        self.ln(h)

    def title_block(self, title, authors_line):
        self.set_f(self.F_TITLE)
        self.multi_cell(self.W(), 8, title, align="C")
        self.gap(2)
        self.set_f(self.F_AUTHOR)
        self.cell(0, 5, authors_line, align="C")
        self.ln(6)
        self.hr(0.5)

    def h1(self, txt):
        self.gap(3)
        self.set_f(self.F_H1)
        self.multi_cell(self.W(), 6.5, txt)
        self.gap(1.5)

    def h2(self, txt):
        self.gap(2)
        self.set_f(self.F_H2)
        self.multi_cell(self.W(), 5.5, txt)
        self.gap(1)

    def h3(self, txt):
        self.gap(1.5)
        self.set_f(self.F_H3)
        self.multi_cell(self.W(), 5.0, txt)
        self.gap(0.5)

    def para(self, txt):
        self.set_f(self.F_REG)
        # Inline simple markdown: **bold**, *italic*
        self._render_inline(txt, line_h=4.6)
        self.gap(2)

    def _render_inline(self, text, line_h=4.6):
        """Tokenise and emit runs with regular/bold/italic font.  Uses
        write() so runs flow within a single justified-ish paragraph."""
        # Sanitize glyphs Arial doesn't have
        replacements = {
            "⟨": "<",  "⟩": ">",
            "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
            "⁰": "^0", "⁻": "^-", "¹": "^1", "²": "^2", "³": "^3",
            "∈": " in ",
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        # Split on **bold** and *italic*
        tokens = re.split(r'(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)', text)
        for tok in tokens:
            if not tok:
                continue
            if tok.startswith("**") and tok.endswith("**"):
                self.set_f(self.F_BOLD); self.write(line_h, tok[2:-2])
            elif tok.startswith("*") and tok.endswith("*") and len(tok) > 2:
                self.set_f(self.F_ITAL); self.write(line_h, tok[1:-1])
            elif tok.startswith("`") and tok.endswith("`"):
                # Courier is a built-in core font (ASCII only) — strip non-ASCII
                clean = tok[1:-1].encode('ascii', errors='replace').decode('ascii')
                self.set_font("Courier", "", 9.0); self.write(line_h, clean)
            else:
                self.set_f(self.F_REG); self.write(line_h, tok)
        self.set_f(self.F_REG)
        self.ln(line_h)

    def bullets(self, items):
        self.set_f(self.F_REG)
        for it in items:
            self.set_x(self.l_margin + 4)
            self.cell(4, 4.6, "•")
            # render the rest using write so inline bold/italic work
            self.set_x(self.l_margin + 8)
            # use multi_cell-like wrapping by writing manually
            self._render_inline(it, line_h=4.6)
        self.gap(1)

    def numbered(self, items):
        self.set_f(self.F_REG)
        for i, it in enumerate(items, 1):
            self.set_x(self.l_margin + 4)
            self.cell(7, 4.6, f"{i}.")
            self.set_x(self.l_margin + 11)
            self._render_inline(it, line_h=4.6)
        self.gap(1)

    def figure(self, path, caption, max_w_frac=0.92):
        """Insert image scaled to page width (or below it), with caption."""
        if not os.path.exists(path):
            self.para(f"[missing: {path}]"); return
        with Image.open(path) as im:
            iw, ih = im.size
        target_w = self.W() * max_w_frac
        target_h = target_w * ih / iw
        # Keep figure on same page where possible
        avail = self.h - self.b_margin - self.get_y()
        if avail < target_h + 12:  # need ~12mm for caption
            self.add_page()
        x = self.l_margin + (self.W() - target_w) / 2
        self.image(path, x=x, w=target_w, h=target_h)
        self.gap(1.2)
        self.set_f(self.F_CAPTION)
        self.multi_cell(self.W(), 4.2, caption, align="C")
        self.gap(2)

    def table(self, header, rows, col_widths=None):
        if col_widths is None:
            col_widths = [self.W() / len(header)] * len(header)
        # Header
        self.set_f(self.F_TABLE_BOLD)
        self.set_fill_color(225, 230, 240)
        for w, h in zip(col_widths, header):
            self.cell(w, 6.0, str(h), border=1, fill=True, align="C")
        self.ln(6.0)
        # Body
        self.set_f(self.F_TABLE)
        for r in rows:
            for w, c in zip(col_widths, r):
                self.cell(w, 5.5, str(c), border=1, align="C")
            self.ln(5.5)
        self.gap(2)

    def displaymath(self, txt):
        """Centered italic display equation. Uses Arial italic (Unicode-capable)."""
        # Sanitize same as inline
        replacements = {
            "⟨": "<", "⟩": ">",
            "₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4",
            "⁰": "^0", "⁻": "^-", "¹": "^1", "²": "^2", "³": "^3",
            "∈": " in ",
        }
        for k, v in replacements.items():
            txt = txt.replace(k, v)
        self.gap(1)
        self.set_font("Arial", "I", 11)
        self.set_x(self.l_margin)
        self.multi_cell(self.W(), 5.5, "    " + txt, align="L")
        self.set_f(self.F_REG)
        self.gap(1.5)


# ============================================================
# BUILD THE PAPER
# ============================================================
pdf = Paper(orientation="P", unit="mm", format="A4")
pdf.set_margins(Paper.L_MARGIN, Paper.T_MARGIN, Paper.R_MARGIN)
pdf.set_auto_page_break(auto=True, margin=Paper.B_MARGIN)

# Register Unicode-capable Arial faces
ARIAL = "/System/Library/Fonts/Supplemental"
pdf.add_font("Arial", "",  f"{ARIAL}/Arial.ttf")
pdf.add_font("Arial", "B", f"{ARIAL}/Arial Bold.ttf")
pdf.add_font("Arial", "I", f"{ARIAL}/Arial Italic.ttf")
pdf.add_font("Arial", "BI",f"{ARIAL}/Arial Bold Italic.ttf")
pdf.add_page()

# ── Title ──
pdf.title_block(
    "When Encoders Hurt: Diagnosing and Stabilising Late-Training Instability "
    "in Graph-Augmented Soft Q-Mixing",
    "Anonymous Authors  —  NeurIPS 2026 Workshop submission (under review)",
)

# ── Abstract ──
pdf.h2("Abstract")
pdf.para(
    "Graph attention encoders are a routine component of value-decomposition methods for cooperative "
    "multi-agent reinforcement learning, where they are presumed to inject inductive biases that help "
    "agents share information.  We report a previously undocumented failure mode that arises when one "
    "such encoder (a two-layer entity / cross-agent graph attention block) is bolted onto Soft-QMIX, "
    "a recent state-of-the-art entropy-regularised value-decomposition algorithm.  After approximately "
    "6M environment steps on SMACv2 Protoss 5v5, the augmented system stops trying to *win* and "
    "instead learns to *survive*: episode length grows by 66%, the number of allies that die per "
    "episode falls by 19%, and test-time win-rate regresses from a 0.79 peak to 0.69.  We trace the "
    "mechanism analytically — the entropy bonus in the Soft-QMIX target inflates with episode length, "
    "and asymmetric per-agent uncertainty induced by the encoder turns this inflation into an "
    "attractive basin around stalling — and confirm it experimentally with a controlled "
    "per-agent-α premise test that produces the same signature in extreme form (test win rate "
    "collapses to 0.51).  We then propose three lightweight stabilising components: (i) decoupling "
    "the per-agent α used in policy sampling from the *uniform* α used in the entropy bonus of "
    "the target, (ii) a Q-spread per-agent sensor that is 3.4× more discriminative than a previously "
    "tried InfoNCE coordination signal, and (iii) a curriculum that fades heterogeneous α in only "
    "after a uniform-α scalar warmup.  The combined system removes the late-training regression "
    "and matches a re-implemented pure Soft-QMIX baseline (last-100-checkpoint test win rate: "
    "0.731 vs 0.729).  We do **not** claim improvement over pure Soft-QMIX; we contribute "
    "(a) a clean characterisation of an instability mode that practitioners adding encoders to "
    "Soft-style methods will likely encounter, (b) a working remediation, and (c) a documented "
    "negative result on InfoNCE-based per-agent coordination sensing in role-randomised SMACv2."
)

# ── 1. Introduction ──
pdf.h1("1. Introduction")
pdf.para(
    "Cooperative multi-agent reinforcement learning (MARL) has converged on a small set of "
    "structurally similar building blocks: a per-agent recurrent encoder, a value-decomposition "
    "mixer (VDN, QMIX, QPLEX), and a centralised-training / decentralised-execution learner "
    "operating on the joint action–value.  Recent work has decorated this stack along several axes: "
    "stronger encoders (graph attention, transformers, attention-over-entities), stronger objectives "
    "(entropy regularisation, contrastive auxiliaries), and explicit per-agent or per-role heterogeneity."
)
pdf.para(
    "These additions are typically presented as *Pareto-improvements* — better representation, "
    "better objective, no harm done.  We show this presumption is wrong for at least one combination "
    "that is natural to try: a graph attention encoder coupled to Soft-QMIX.  The combination *does* "
    "learn faster early in training and peaks higher than the unaugmented baseline, but then "
    "regresses, ending below the baseline at 10M training steps.  The regression has a clean "
    "signature — episode length grows, agents stop dying, fewer enemies die — that distinguishes "
    "it from generic plateau or overfitting."
)
pdf.para(
    "We diagnose the mechanism, replicate it in extreme form by deliberately inducing it (a "
    "per-agent fixed-α variant collapses to 0.51 win-rate by 10M), and propose three small "
    "modifications that remove it.  The stabilised system matches the unaugmented baseline; "
    "importantly, it does *not* exceed it.  We do not claim a state-of-the-art result.  We "
    "claim a useful *negative* result and a useful *characterisation*: practitioners who add "
    "encoders to entropy-regularised value-decomposition methods can expect this failure mode, "
    "and the remediation is mechanically simple."
)
pdf.h3("Contributions")
pdf.numbered([
    "We document a previously unreported late-training stalling phenomenon in encoder-augmented "
    "Soft-QMIX (§3) and identify the entropy-bonus accumulation mechanism (§4) that drives it.",
    "We confirm the mechanism via a controlled per-agent-α premise test that reproduces stalling "
    "in extreme form (§4.3).",
    "We propose three remediation components — decoupled sampling-vs-target α, a Q-spread-based "
    "per-agent sensor, and a uniform-to-heterogeneous curriculum (§5) — and show they jointly "
    "remove the regression while leaving asymptotic performance intact (§6).",
    "We report a documented negative result: an InfoNCE-based per-agent coordination signal "
    "fails to differentiate agents in role-randomised SMACv2 by a factor of 3.4× compared to "
    "our simpler Q-spread sensor (§6.4).",
])

# ── 2. Background ──
pdf.h1("2. Background")
pdf.h2("2.1 Cooperative MARL and value decomposition")
pdf.para(
    "We consider Dec-POMDPs ⟨S, A, O, P, R, n, γ⟩ with n cooperative agents sharing reward r_t.  "
    "Each agent i holds a partial observation history τ_i and outputs a per-agent action–value "
    "Q_i(τ_i, ·).  Value-decomposition methods approximate the joint Q_tot as a monotone "
    "combination of per-agent Q_i, enabling decentralised execution from centralised training."
)
pdf.h2("2.2 Soft-QMIX")
pdf.para(
    "Soft-QMIX is a recent entropy-regularised variant.  The mixer combines a VDN-like sum with "
    "two learned operators: g(·), an order-preserving residual that shapes per-agent Q values "
    "without changing their action ranking, and f(·), an affine per-agent operator (a learned "
    "per-state temperature)."
)
pdf.para("Action selection during training samples from the soft policy")
pdf.displaymath("π_i(a | s)  =  softmax( f_i( g_i( Q_i(s, ·) ) ) / α )")
pdf.para("and the TD(λ) target uses a sample-based entropy bonus,")
pdf.displaymath("y_t  =  r_t + γ ( Q_tot^tgt(s_{t+1}, a*_{t+1})  −  α · Σ_i log π_i(a*_{t+1,i} | s_{t+1}) )")
pdf.para(
    "where a*_{t+1} ~ π is sampled from the *online* policy (a Double-Q-style choice) and "
    "evaluated under the *target* network.  The scalar α is the entropy coefficient; in the "
    "original formulation α ≈ 0.03 is shared by all agents."
)
pdf.h2("2.3 Graph attention encoders for MARL")
pdf.para(
    "A common upgrade to the recurrent encoder is a two-stage graph attention block: "
    "(1) **Local entity attention** — each agent attends over its own observable entities "
    "(own unit, visible enemies, visible allies), producing a per-agent local summary ℓ_i; "
    "(2) **Cross-agent attention (TeamGAT)** — agents attend over each other's local summaries, "
    "producing a contextualised hidden state h_i.  In our implementation, ℓ_i and h_i both have "
    "dimension 128 with 4 attention heads, layer norm, and orthogonal initialisation.  The "
    "Q-head is a single linear layer on h_i."
)

# ── 3. The Stalling Phenomenon ──
pdf.h1("3. The Stalling Phenomenon")
pdf.h2("3.1 Setup")
pdf.para(
    "We compare four configurations on SMACv2 Protoss 5v5.  All runs use 8 parallel environments, "
    "replay buffer 5000 episodes, batch size 128, learning rate 10⁻³, γ = 0.99, λ = 0.4, hard "
    "target updates every 200 episodes, 10M environment steps total.  All runs are single-seed "
    "and use the same environment seed schedule."
)
pdf.table(
    header=["Run", "Encoder", "α regime"],
    rows=[
        ["Pure Soft-QMIX", "RNN (hidden 64)", "scalar α=0.03"],
        ["Run A", "GAT (hidden 128, 4 heads)", "scalar α=0.03"],
        ["Run B", "GAT (hidden 128, 4 heads)", "per-agent fixed [0.01..0.05]"],
        ["Run C", "GAT (hidden 128, 4 heads)", "proposed (§5)"],
    ],
    col_widths=[35, 65, 70],
)
pdf.h2("3.2 The empirical signature")
pdf.para(
    "Figure 2 shows the stalling signature.  In Run A, between 6M and 10M environment steps:"
)
pdf.bullets([
    "**Test win rate** falls from a rolling-30 MA peak of 0.789 (at t = 6.07 M) to 0.686 "
    "(last-100 mean).  The fall is monotone per 1M-step bin: 0.748 → 0.722 → 0.686.",
    "**Test episode length** grows from 94.5 (4–5 M) to 109.8 (9–10 M).  Pure Soft-QMIX in "
    "the same window grows only from 79 to 87.",
    "**Dead-allies-per-episode** falls from 3.58 (4–5 M) to 3.30 (9–10 M).  The team is *less* "
    "likely to have ally casualties as training progresses.",
    "**Dead-enemies-per-episode** stays roughly constant at 4.3–4.5.  Combat does not produce "
    "more outcomes — games time out.",
    "Q-value mean continues to grow (1.22 → 1.55), loss continues to fall, gradient norm stays "
    "bounded around 2.0.  The optimiser is healthy.  The optimiser is just chasing a different "
    "objective than 'win the game'.",
])

pdf.figure(os.path.join(HERE, "fig_main_result.png"),
           "Figure 1: SMACv2 Protoss 5v5, single seed, 10M steps. Run A peaks higher early "
           "but regresses; Pure Soft-QMIX is still climbing at 10M and ends near Run C.")

pdf.figure(os.path.join(HERE, "fig_stalling.png"),
           "Figure 2: The stalling signature in Run A — and its absence in Run C and Pure SQ. "
           "Run A's episode length grows while dead allies fall and win rate regresses. "
           "Run C and Pure SQ both avoid this trajectory.")

pdf.para(
    "The key observation is that Run A's regression begins *after* the policy has already "
    "learned to win the majority of games.  The team finds a 'win 79% by committing' basin, "
    "then drifts out of it into a 'stall and survive' basin."
)

# ── 4. Mechanistic Analysis ──
pdf.h1("4. Mechanistic Analysis")
pdf.h2("4.1 The regularised objective")
pdf.para("The optimiser maximises the discounted return *plus* a per-agent entropy bonus:")
pdf.displaymath("J(π)  =  E_π [ Σ_t γ^t ( r_t + α · Σ_i H( π_i(· | s_t) ) ) ]")
pdf.para(
    "For a T-step trajectory with per-agent average entropy H̄, the cumulative entropy bonus "
    "contribution to J is approximately"
)
pdf.displaymath("B(T)  ≈  α · n · T · H̄ · (1 − γ^T) / (1 − γ)")
pdf.para(
    "That is: B scales with episode length.  A policy that produces *long* episodes harvests "
    "more entropy bonus than one producing short episodes, all else equal.  This is the "
    "structural pressure toward stalling."
)
pdf.para(
    "In standard actor–critic settings this pressure is benign because (a) the reward signal "
    "grows with episode length too (more time = more chances to score), and (b) all agents "
    "share the same α and the same exploration budget.  The regularisation is symmetric and "
    "the team's joint policy still aligns with reward."
)
pdf.h2("4.2 Why an encoder breaks the symmetry")
pdf.para(
    "A graph attention encoder produces per-agent hidden states h_i whose pairwise cosine "
    "similarity is empirically *not* uniform.  We observe (Run A, 5–6 M window):"
)
pdf.bullets([
    "cos(h_i, h_j) = 0.71 on average — agents share substantial direction.",
    "cos(ℓ_i, ℓ_j) = 0.67 at the local-summary stage.",
    "cos(Δℓ_i, Δℓ_j) = 0.04 on time deltas — orthogonal.",
])
pdf.para(
    "The encoder produces homogenised hidden states in *space* but heterogeneous trajectories "
    "in *time*.  The downstream Q-head then produces per-agent Q-vectors with *different* "
    "peakedness per agent — i.e., the *effective* exploration temperature α / spread(Q_i) "
    "varies across agents even when the nominal α is shared.  The regularisation has become "
    "silently per-agent."
)
pdf.para(
    "The optimiser, faced with asymmetric per-agent regularisation, finds an asymmetric "
    "solution.  In a cooperative team, the simplest such solution is 'the most-explorative "
    "agent disrupts coordinated offensive plays, so the team learns plays that don't require "
    "that agent to commit', which *is* stalling."
)
pdf.h2("4.3 The premise test (Run B)")
pdf.para(
    "To confirm this mechanism, we ran a controlled experiment: deliberately introduce per-agent "
    "α by hand and observe the same signature in extreme form.  We set α_i = 0.01 + 0.01·i for "
    "i ∈ {0,1,2,3,4} — same mean as Run A (0.03), 5× spread."
)
pdf.para(
    "Result (Figure 3): the stalling phenomenon amplifies dramatically.  Episode length grows "
    "from 95 (2–3 M) to 134 (9–10 M).  Dead allies fall from 3.60 to 2.99.  Test win rate peaks "
    "at 0.695 (rolling-30 MA, 5.7 M) then collapses to 0.512 (last-50 mean).  The InfoNCE "
    "coordination signal we had attempted to use for adaptive α also fails to differentiate "
    "agents (coord_signal_std ≈ 0.003 throughout, despite explicit 5× α heterogeneity)."
)
pdf.figure(os.path.join(HERE, "fig_runB_diagnosis.png"),
           "Figure 3: Per-agent α heterogeneity (Run B, purple) catastrophically amplifies the "
           "late-training stalling phenomenon already present in milder form in Run A (red).")
pdf.para(
    "This confirms the mechanism: explicit α heterogeneity in both sampling *and* target "
    "produces the stalling attractor by structurally privileging long episodes for high-α "
    "agents.  The encoder-induced *implicit* α heterogeneity in Run A produces the same "
    "signature in milder form."
)

# ── 5. Proposed Stabilisation ──
pdf.h1("5. Proposed Stabilisation")
pdf.h2("5.1 Component 1: Decouple sampling-α from target-α")
pdf.para("In standard Soft-QMIX, the same α appears in two structurally different places:")
pdf.bullets([
    "**Policy sampling**: π_i(a|s) = softmax( f_i(g_i(Q_i)) / α_i ) — controls *exploration*.",
    "**Target entropy bonus**: −α_i · log π_i(a*_i | s) — controls the *objective*.",
])
pdf.para(
    "Component 1 makes these explicit and breaks them apart.  We use an arbitrary per-agent "
    "α_i in the *sampling* step (heterogeneous exploration) but the **scalar mean** "
    "ᾱ = (1/n)·Σ_i α_i uniformly in the target:"
)
pdf.displaymath("y_t  =  r_t + γ ( Q_tot^tgt(s_{t+1}, a*_{t+1})  −  ᾱ · Σ_i log π_i(a*_{t+1,i} | s_{t+1}) )")
pdf.para(
    "This makes the regularised *objective* identical to scalar-α Soft-QMIX (no asymmetric "
    "stalling pressure) while preserving heterogeneity in the *exploration distribution*.  "
    "Mathematically, when all α_i = ᾱ it reduces to standard Soft-QMIX.  Component 1 is a "
    "strict generalisation."
)
pdf.h2("5.2 Component 2: Q-spread per-agent sensor")
pdf.para(
    "We need a per-agent signal to drive α_i.  Prior work (our previous attempt, see §6.4) used "
    "InfoNCE on encoder hidden states; this fails because role randomisation in SMACv2 prevents "
    "stable per-agent identity from emerging in the encoder.  Instead, we use the *spread* of "
    "an agent's own Q-vector,"
)
pdf.displaymath("s_i(t)  =  max_a Q_i(s_t, a)  −  min_a Q_i(s_t, a),")
pdf.para(
    "aggregated as an EMA ŝ_i with τ = 0.99.  Wide ŝ_i means the agent has a clearly best "
    "action — it should commit (low α_i).  Narrow ŝ_i means Q values are flat — it should "
    "hedge (high α_i).  The per-agent confidence is c_i = ŝ_i / mean(ŝ), clipped to [0.3, 3.0] "
    "to prevent degenerate agents, and α_i = ᾱ / c_i."
)
pdf.para(
    "Q-spread is computed every train step from quantities the learner already produces; it "
    "requires no additional network, no separate optimiser, and no auxiliary loss."
)
pdf.h2("5.3 Component 3: Curriculum from uniform to heterogeneous")
pdf.para(
    "Heterogeneous α must not disrupt early-training coordination.  We linearly ramp it in: a "
    "ramp factor ρ(t) is 0 before t₀ = 2M, 1 after t₁ = 4M, and linear in between.  The "
    "effective α is"
)
pdf.displaymath("α_i^eff(t)  =  ᾱ + ρ(t) · ( α_i − ᾱ )")
pdf.para(
    "For t < t₀, the system runs as pure scalar-α Soft-QMIX; over [t₀, t₁] it phases in "
    "heterogeneity; for t > t₁ it runs at full heterogeneity.  The motivation is empirical: in "
    "Run B, heterogeneity from step zero produced the catastrophic stalling; we expect to need "
    "symmetric foundations before introducing asymmetry."
)
pdf.figure(os.path.join(HERE, "fig_components.png"),
           "Figure 4: Components 2 and 3 in action.  Left: curriculum ramp ρ(t) and resulting "
           "α heterogeneity (α_std).  Right: per-agent effective α evolves into a stable "
           "ranking after curriculum end (4M).")

# ── 6. Experiments ──
pdf.h1("6. Experiments")
pdf.h2("6.1 Setup")
pdf.para(
    "All runs use SMACv2 Protoss 5v5, 10M environment steps, single seed.  We compare Pure "
    "Soft-QMIX (RNN encoder, scalar α; current SOTA baseline), Run A (GAT encoder, scalar α; "
    "encoder augmentation only), Run B (GAT encoder, per-agent fixed α; the failure case), "
    "and Run C (GAT encoder + Components 1+2+3; proposed stabilisation).  We report rolling-30 "
    "moving averages of test win rate, with raw test checkpoints scattered for noise visibility."
)

pdf.h2("6.2 Main result")
pdf.para("Quantitative summary of the four runs:")
pdf.table(
    header=["Metric", "Pure SQ", "Run A", "Run B", "Run C"],
    rows=[
        ["last-100 mean",       "0.729",  "0.686",  "0.514",  "0.731"],
        ["last-50 mean",        "0.747",  "0.666",  "0.475",  "0.722"],
        ["rolling-30 MA peak",  "0.766",  "0.789",  "0.695",  "0.768"],
        ["ep_length (9–10 M)",  "86.98",  "109.82", "133.71", "98.73"],
        ["dead_allies (9–10 M)","3.215",  "3.304",  "2.988",  "3.543"],
    ],
    col_widths=[55, 30, 30, 30, 30],
)
pdf.h3("Key observations")
pdf.numbered([
    "**Run A regresses by 17.8% relative** between its 4–7 M peak window (0.625 mean) and its "
    "8–10 M end window (0.514 mean).  Run C and Pure SQ do not regress.",
    "**Run C's Components remove the regression**.  Last-100 mean climbs from Run A's 0.686 to "
    "Run C's 0.731 (+4.5 pp).  Episode length growth attenuates from +43 steps (Run A) to +33 "
    "steps (Run C).  Dead-allies stays high (3.54 vs Run A's 3.30).",
    "**Run C does not exceed Pure Soft-QMIX**.  Last-100 means are within 0.002 of each other "
    "(0.731 vs 0.729).  Pure SQ's last-50 actually exceeds Run C (0.747 vs 0.722).  We do not "
    "claim improvement over the unaugmented baseline.",
    "**Run B confirms the mechanism**.  By forcing the same α heterogeneity Run A produced "
    "silently, we obtained a clearly worse outcome — same direction as Run A's regression but "
    "in extreme form (last-50 mean 0.475 vs Run A's 0.666, –19 pp).",
])

pdf.h2("6.3 The stalling diagnostic")
pdf.para(
    "Figure 2 (above) shows the stall metrics across the three runs comparable on the chart "
    "(A, C, Pure SQ).  Run A's curves follow the stalling signature (ep_length grows, "
    "dead_allies falls).  Run C's curves are flatter.  Pure SQ's are flattest.  The shapes of "
    "these curves — not just the absolute numbers — distinguish the failure mode from generic "
    "plateau or overfitting."
)

pdf.h2("6.4 Sensor comparison: Q-spread vs InfoNCE coord_signal")
pdf.para(
    "Our Q-spread sensor (Component 2) is a replacement for an earlier InfoNCE-based "
    "coordination sensor that we attempted to use.  The InfoNCE sensor projected each agent's "
    "hidden state through a two-layer MLP and predicted *teammate* hidden state deltas via a "
    "contrastive objective with K cross-episode negatives.  Despite three architectural "
    "iterations (raw h_i targets, pairwise loss, identity-conditioned predictor), the "
    "per-agent coord_signal_std never exceeded 0.005.  In Run B — which has *explicit* 5× α "
    "differentiation — the InfoNCE signal still failed to differentiate agents (Figure 5, "
    "dashed line: σ_coord ≈ 0.003)."
)
pdf.para(
    "The Q-spread sensor (Figure 5, solid green) maintains σ_Q-spread ≈ 0.012 across the same "
    "window — **3.4× more discriminative**.  Because Q-spread is computed from quantities the "
    "learner produces anyway, it has no additional parameters or compute, and it captures "
    "*state-conditioned* uncertainty (which the encoder's static embeddings cannot, in "
    "role-randomised tasks)."
)
pdf.figure(os.path.join(HERE, "fig_sensors.png"),
           "Figure 5: The Q-spread sensor (solid green, Component 2) is 3.4× more discriminative "
           "than the InfoNCE coord_signal we previously tried (dashed grey) for per-agent "
           "differentiation in role-randomised SMACv2.")
pdf.para(
    "This is a documented negative result.  InfoNCE-based per-agent coordination signals — "
    "proposed in several recent papers as a way to drive role-aware exploration — fail in "
    "environments where role assignments shuffle each episode.  Practitioners should not "
    "expect cross-agent identity to emerge from contrastive learning over shared encoder "
    "representations when there is no consistent identity to extract."
)

# ── 7. Limitations ──
pdf.h1("7. Limitations")
pdf.para("We are explicit about what this paper does and does not establish.")
pdf.bullets([
    "**Single seed.**  All runs are single-seed.  The within-run rolling-30 standard deviation "
    "is approximately 0.08 in win-rate.  Some of the gaps we report (Run A → Run C: +4.5 pp on "
    "last-100) are roughly 1σ.  We cannot make significance claims and we do not.  Run B's "
    "collapse (–22 pp from Run A) is well outside seed noise and is the cleanest result in the "
    "paper.",
    "**Single environment.**  SMACv2 Protoss 5v5 only.  We do not claim the phenomenon or the "
    "fix generalise to Terran/Zerg, larger team sizes, or other cooperative MARL benchmarks.  "
    "We *suspect* the mechanism is general (the math in §4.1 does not depend on the "
    "environment), but suspecting is not knowing.",
    "**No new SOTA.**  Run C matches Pure Soft-QMIX, it does not exceed it.  The contribution "
    "is a characterisation and remediation of an instability mode, not a new high-water mark.",
    "**Limited ablation.**  We did not run Component 1 alone or Component 2 alone.  We cannot "
    "quantitatively partition Run C's improvement over Run A across the three components.",
    "**Pure Soft-QMIX is still climbing at 10M.**  Its terminal performance is a moving target.  "
    "A 15M training budget might place it materially above Run C; we do not know.",
])

# ── 8. Related Work ──
pdf.h1("8. Related Work")
pdf.bullets([
    "**Soft-style value decomposition.**  Soft-QMIX adapts entropy-regularised Q-learning to "
    "value-decomposition.  SMIX uses softmax operators in the Bellman target.  Our work studies "
    "*stability* of these methods under encoder augmentation, an angle not investigated by "
    "either.",
    "**Encoder augmentation in MARL.**  GATs, transformers, attention-over-entities, and HARL "
    "all add structured encoders.  None of these papers report or study late-training "
    "instability; they typically report monotonic-looking learning curves to a fixed budget.",
    "**Per-agent / role-aware MARL.**  ROMA learns explicit roles; CDS uses contrastive role "
    "discovery; SDC uses identity-conditioned policies.  We tried InfoNCE-based role discovery "
    "(§6.4) and document its failure on role-randomised SMACv2.",
    "**Stalling / non-completion in cooperative RL.**  Stalling has been reported anecdotally "
    "in cooperative tasks where avoiding death has a higher reward gradient than achieving "
    "objective.  Our contribution is to (a) connect this to entropy regularisation specifically, "
    "(b) derive its dependence on episode length, and (c) propose a structural fix that does "
    "not require reward shaping.",
])

# ── 9. Discussion and Conclusion ──
pdf.h1("9. Discussion and Conclusion")
pdf.para(
    "We have characterised a previously unreported failure mode in the increasingly common "
    "practice of bolting structured encoders onto entropy-regularised value-decomposition "
    "methods.  The mechanism — entropy bonus accumulation under asymmetric per-agent "
    "uncertainty — is general enough that we expect similar failures in transformer + Soft-QMIX, "
    "GAT + SQDDPG, and other natural combinations.  The signature — episode length growth + "
    "ally preservation + win rate regression — is easy to look for in published learning "
    "curves, and we suspect it is present, unannotated, in a number of recent results."
)
pdf.para(
    "Our remediation is structural and lightweight: decouple where α appears in the target "
    "from where it appears in sampling, drive heterogeneity from a sensor that uses "
    "information already on the learner's tape, and warm up before introducing asymmetry.  "
    "The combined system removes the regression.  It does not produce a new SOTA on the task "
    "we tested, and we do not present it as one."
)
pdf.h3("Future work")
pdf.numbered([
    "**Apply the three components to Pure Soft-QMIX (no encoder)** — the test that determines "
    "whether Components 1–3 are a fundamental contribution or merely a fix for encoder-induced "
    "damage.",
    "**Multi-seed validation** (3–5 seeds per condition) to make all comparisons statistically "
    "defensible.",
    "**Cross-scenario validation** on at least one other SMACv2 variant.",
])
pdf.para(
    "Until these are done, this work is — we think appropriately — a workshop submission, "
    "not a main-conference one."
)

# ── Reproducibility ──
pdf.h1("Reproducibility Statement")
pdf.para(
    "All code is built on top of pymarl2 with SMACv2; configurations are provided as YAML "
    "files.  Components 1–3 add approximately 60 lines to the learner and one new field to "
    "the action selector.  Hyperparameters: ᾱ = 0.03, τ_Q-spread = 0.99, c ∈ [0.3, 3.0], "
    "t₀ = 2M, t₁ = 4M.  Training uses a single GPU; total wall-clock per 10M-step run is "
    "approximately 16 hours."
)

# ── Save ──
out = os.path.join(HERE, "..", "paper_final.pdf")
pdf.output(out)
print(f"Wrote {os.path.abspath(out)}")
print(f"  size: {os.path.getsize(out):,} bytes  pages: {pdf.page_no()}")
