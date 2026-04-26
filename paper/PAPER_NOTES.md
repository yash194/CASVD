# Paper writeup — submission notes

## Files in `paper/`

- `paper.md` — full paper draft, ~7 pages. Structured sections, ready to convert to LaTeX.
- `fig_main_result.png` — Figure 1 (hero figure: 3-way head-to-head).
- `fig_stalling.png` — Figure 2 (stall signature: win rate / ep_len / dead allies / dead enemies).
- `fig_runB_diagnosis.png` — Figure 3 (Run B premise test, catastrophic stalling).
- `fig_components.png` — Figure 4 (Components 2 and 3 in action: ramp + α evolution).
- `fig_sensors.png` — Figure 5 (Q-spread vs InfoNCE coord_signal — the sensor negative result).

## Where it stands honestly

**Workshop tier (NeurIPS / ICML workshops):** strong probability of acceptance if framed as **diagnosis + remediation paper**.
**Main conference:** not yet — would need (a) Pure SQ + Components 1+2+3 run, (b) multi-seed, (c) second SMACv2 scenario.

## What the paper claims

1. *We document* — a previously unreported late-training stalling phenomenon in encoder-augmented Soft-QMIX.
2. *We diagnose* — entropy-bonus accumulation under asymmetric per-agent uncertainty.
3. *We remediate* — three lightweight components remove the regression.
4. *We document a negative result* — InfoNCE per-agent coordination signals fail in role-randomised SMACv2.

## What the paper does NOT claim

- Improvement over the unaugmented Pure Soft-QMIX baseline (it matches, doesn't exceed).
- Statistical significance (single-seed; we are explicit).
- Generalisation across SMACv2 variants (only Protoss 5v5 tested).
- A new state-of-the-art.

This honesty is *load-bearing*. Workshop reviewers reward careful negative results and clear failure-mode characterisation; they punish overclaims. The paper should be submitted with the §7 (Limitations) section intact.

## Recommended target venues

In rough order of fit:

1. **NeurIPS Workshop on Cooperative AI** — focuses on multi-agent coordination; loves diagnostic papers and negative results.
2. **NeurIPS Workshop on RL: Algorithms and Applications** — broad RL workshop, methods + diagnostic stories welcome.
3. **AAMAS Workshop on Multi-Agent Sequential Decision Making (MSDM)** — direct topic match.
4. **ICML Workshop on RL Theory** — if you want to lean into the §4.1 mechanistic analysis.

## Strongest pitches per audience

- **Methods-oriented audience**: the §5.1 decoupling argument. "Soft-QMIX has a hidden assumption — that α is shared by all agents — and that assumption breaks the moment you augment the encoder. We show the failure and the fix is a one-line code change."
- **Empirical-MARL audience**: the §3.2 stalling signature. "Watch for ep_length growing while dead_allies falls. We provide a checklist."
- **Negative-results-loving audience**: the §6.4 InfoNCE failure. "Don't use InfoNCE on shared encoders in role-randomised tasks. Here's why, with three architectural attempts."

## Required additional work before submission

In priority order (~2 weeks total):

1. **Run Pure Soft-QMIX + Components 1+2+3** (1 day GPU time).
   - If it beats Pure Soft-QMIX cleanly: paper upgrades from workshop-tier to main-conference-tier.
   - If it ties: paper stays workshop-tier but is more honest.
2. **Run 2 more seeds of Run A and Run C** (~4 days).
3. **Expand Section 8 references** properly cite Soft-QMIX, QMIX, GAT, SMACv2, ROMA, CDS papers.
4. **Convert paper.md to LaTeX** using the workshop's template; usually 2–3 hours.

## What to fix if a reviewer pushes back

- "Single seed!" → "Yes — see Limitations §7. Run B's catastrophic collapse (–22 pp) is well outside any plausible seed variance, so the *mechanism* claim stands. The *magnitude* of Components' contribution requires multi-seed."
- "You're not beating SOTA!" → "Correct — see Limitations §7. We position this as a diagnostic + stabilisation paper. The contribution is the failure mode characterisation and the three-component fix that demonstrably removes it."
- "Why did you use GAT in the first place?" → "We explored encoder augmentation as a presumed Pareto-improvement and discovered the failure mode through that exploration. The paper itself argues practitioners should be cautious about adding encoders to entropy-regularised methods."

## Don't oversell

The temptation is to spin Run C's "match" of Pure SQ as a "tie with SOTA." Don't. Reviewers will check the numbers. Be the paper that openly says "we matched, we didn't beat" and reviewers will trust everything else you write.
