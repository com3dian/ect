# Aim 3 — Four-point geometry & triangle stress tests (reframed plan)

## Why this package exists

Earlier Aim-3 chain packages (`aim3_additive_chain_*`, v2–v7) tested whether Type A
phylogenetic chains satisfy **strong path additivity**:

\[
\pi(n_4 \mid n_1) \stackrel{?}{\approx} \pi(n_2 \mid n_1)\,\pi(n_3 \mid n_2)\,\pi(n_4 \mid n_3)
\quad\Leftrightarrow\quad
d(n_1,n_4) \stackrel{?}{\approx} d(n_1,n_2)+d(n_2,n_3)+d(n_3,n_4)
\]

with \(d_M = -\ln\pi\) from the model’s own conditional probabilities.

**Result (Gemma-2-2B and Gemma-2-9B):** that equality does **not** hold (gaps are large
and negative). Triangle inequality usually holds, but that is a weak bar.

**Reframe:** do not ask “is the chain a geodesic?” Ask:

> For concept **quadruples** with known hierarchical structure, do all **six pairwise**
> model distances look **more tree-like** than for scrambled/adversarial quadruples?

The headline claim becomes **relative**: the model **prefers** coherent family order
(smaller slack / better four-point score), not perfect tropical additivity.

Type **B** and **C** stay as **triangle stress tests** under disruption and framing.
This README is the plan for implementing that unified story in one place.

---

## Shared metric

For prefix \(x\) and target string \(y\) (concept label or continuation):

\[
d_M(x,y) = -\sum_i \log \pi(t_i \mid x, t_{<i})
\]

Use **sequence log-probability** (sum of token logprobs), not mean logprob or
fingerprint overlap. Same cloze template family within each experiment type.

**Corpus:** `src/aim3_data_prep/data/tropical_geometry_corpus.json`  
**Models:** Gemma-2-2B is the **default on this cluster** (`output_gemma-2-2b/`).  
Gemma-2-9B runs on another server when ready (`output_gemma-2-9b/`).

---

## Experiment 1 — Type A: four-point condition on quadruples

### Data

Each Type A item is a quadruple \((n_1,n_2,n_3,n_4)\) with intended hierarchy, e.g.

`Animal → Mammal → Dog → Poodle`.

**Controls** (same four concepts, permuted order):

| Variant | Example order | Role |
|---|---|---|
| `natural` | \(n_1 \to n_2 \to n_3 \to n_4\) | true phylogenetic chain |
| `shuffled_middles` | swap middle two | breaks taxonomy, keeps endpoints |
| `reversed` | \(n_4 \to n_3 \to n_2 \to n_1\) | wrong direction |
| (optional) `random_labels` | unrelated concepts | adversarial baseline |

### Six pairwise distances

For each variant, compute **all six** directed conditionals with a **fixed cloze
template** (one design per pair type, applied consistently):

\[
d(n_i, n_j) = d_M\bigl(\text{cloze}(n_i),\, n_j\bigr)
\]

for every unordered pair \(\{i,j\}\), \(i \neq j\) (store both directions or one
canonical direction — pick one rule and stick to it).

Do **not** mix unrelated prompt styles across pairs within the same item.

### Four-point condition (tree-likeness score)

Label the four nodes \(a,b,c,d\). Form the three pair sums:

\[
S_1 = d(a,b) + d(c,d),\quad
S_2 = d(a,c) + d(b,d),\quad
S_3 = d(a,d) + d(b,c)
\]

In a **tree metric**, the **two largest** of \(\{S_1,S_2,S_3\}\) are equal.

**Per-item score** (locked in code):

\[
\text{fp\_slack} = \max(S_1,S_2,S_3) - \text{second\_max}(S_1,S_2,S_3)
\]

Lower slack ⇒ more tree-like. (We use max − second-max, not max − median.)

**Primary comparison:** `natural` vs `shuffled_middles` (same four concepts, two
orderings) on paired fp_slack across items.

### Statistical inference (Type A primary)

Do **not** use a paired *t*-test on fp_slack — distributions are skewed and
non-negative.

For each item, same quadruple under two orderings:

\[
\Delta_i = \text{fp\_slack}_i^{\text{shuffled}} - \text{fp\_slack}_i^{\text{natural}}
\]

Positive \(\Delta_i\) means natural is more tree-like (lower slack).

| Test | What it asks | Implementation |
|---|---|---|
| **Wilcoxon signed-rank** | Is median \(\Delta > 0\)? | `scipy.stats.wilcoxon(Δ, alternative="greater")` |
| **Paired fraction** | What % of items have natural \(<\) shuffled? | Report `%`; headline number |
| **Binomial / sign test** | Is that fraction \(> 0.5\)? | `scipy.stats.binomtest(k, n, p=0.5, alternative="greater")` |

Both tests are written to `four_point_summary.json` under
`by_type.A.paired_tests.natural_vs_shuffled_middles`. The binomial test formalizes
the headline paired fraction; Wilcoxon uses the full slack magnitudes.

Secondary: same paired machinery for `natural` vs `reversed` (exploratory).

**Claims we can make:**

- Natural quadruples are **more tree-like** than shuffled (expect ~80% paired
  improvement, consistent with prior Markov-gap controls).
- Gaps are **not** near zero; the model does **not** implement exact geodesic
  additivity.

**Claims we should not make:**

- “The model embeds Type A chains as exact tropical geodesics.”
- “Reversed is always worse than natural” (check empirically; v6 showed reversed
  sometimes slightly closer on hop-additivity — four-point may differ).

### Optional secondary metrics (appendix)

- Hop additivity gap: \(d(n_1,n_4) - \sum\) consecutive hops (v6-style).
- Path vs direct on adjacent pairs only.

---

## Experiment 2 — Type B: triangle under narrative disruption

**Story:** normal context → sudden twist → combined sentence.

Example: accountant + spreadsheet → lava portal.

**Three distances:**

1. \(d(\text{base}, \text{shift})\)
2. \(d(\text{shift}, \text{combined})\)
3. \(d(\text{base}, \text{combined})\) — shortcut

**Triangle slack:**

\[
\text{slack}_B = d(\text{base},\text{combined}) - d(\text{base},\text{shift}) - d(\text{shift},\text{combined})
\]

**Question:** when the narrative jumps, does the metric stay consistent (slack
\(\le 0\)), and how large is the shortcut advantage?

**Framing:** stress test under **surprisal / plot twist**, not tree-likeness.

---

## Experiment 3 — Type C: triangle under adversarial framing

**Story:** entailed concept, superordinate, plus a **surreal framed sentence**
that still mentions those concepts.

Example: poodle / dog in a dream-on-Mars frame.

**Three distances** (same cloze family as v6):

1. \(d(\text{entailed}, \text{superordinate})\)
2. \(d(\text{framed}, \text{entailed})\)
3. \(d(\text{framed}, \text{superordinate})\)

**Triangle slack:**

\[
\text{slack}_C = d(e,s) - d(f,e) - d(f,s)
\]

(using shorthand \(e=\) entailed, \(s=\) superordinate, \(f=\) framed text as
conditioning context.)

**Question:** does extreme framing **break** metric consistency more than Type B?

**Prior result (9B):** ~21% violation rate (slack \(> 0\)) vs 0% for Type B — worth
replicating under this unified pipeline.

---

## What we ship (paper / repo narrative)

| Block | Type | Question | Expected headline |
|---|---|---|---|
| **Chain / tree** | A | Four-point on six pair distances | Model **prefers** hierarchical order over shuffles |
| **Disruption** | B | Triangle slack | TI holds; shortcuts dominate after twists |
| **Framing** | C | Triangle slack | More **violations** under surreal frames |

One sentence summary:

> LM conditional log-probabilities define a generalized metric that is **partially
> tree-like on hierarchical quadruples**, **stressed but consistent under narrative
> jumps**, and **more fragile under adversarial framing**.

Ship **all three**; demote “hop gap ≈ 0” to rejected strong hypothesis / appendix.

---

## Implementation sketch (this folder)

Planned layout (implemented):

```
aim3_four_point_geometry/
  README.md                 ← this plan
  config.py
  prompts.py                ← unified cloze templates (A/B/C)
  model_io.py               ← sequence log π (reuse v6 patterns)
  distances.py              ← six pair + four-point; B/C triangle; paired stats
  experiment.py             ← run all types; variants for Type A
  plot_type_a_four_point.py
  plot_type_bc_triangle.py
  run_experiment.slurm
  output_gemma-2-2b/        ← default on this cluster
```

**Reuse:** prompt/metric patterns from `aim3_additive_chain_6`; do not duplicate
SAE/J-lens fingerprint metrics from v2–v5.

**Controls:** always report paired natural vs shuffled (and reversed); include
model slug in output paths (`output_gemma-2-2b`, `output_gemma-2-9b`).

**HF cache:** source `scripts/hf_lustre_env.sh` on SLURM (lustre scratch).

---

## Success criteria (before writing up)

1. **Type A (2B on this cluster):** clear separation natural vs shuffled on
   four-point slack — Wilcoxon \(p \ll 0.05\), paired fraction well above 50%
   with binomial \(p \ll 0.05\). Replicate 9B separately when that run completes.
2. **Type B:** slack \(\le 0\) for almost all items; document mean shortcut margin.
3. **Type C:** violation rate \(>\) Type B; interpret as framing-induced warp.
4. **Honesty check:** no condition has median slack near 0 unless actually measured.

---

## Relation to prior packages

| Package | Role now |
|---|---|
| `aim3_data_prep` | Corpus (unchanged) |
| `aim3_additive_chain_6` | Prototype Markov cloze + controls; hop-additivity **failed** |
| `aim3_additive_chain_7` | Scoring sanity notes (BPE composition); not the main result |
| `aim3_additive_chain_5` | TVD on fingerprints — different metric; appendix only |
| **`aim3_four_point_geometry`** | **Canonical reframed evaluation (this plan)** |

---

## Design choices (locked in code)

1. **Directed pairwise distances** only: \(d(i,j) = d_M(\text{cloze}(n_i), n_j)\) for
   chain labels \(0..3\) after permutation; no symmetrization.
2. **Single cloze stem** for all six pairs: “A more specific kind of that concept is”.
3. **Four-point slack:** max − second_max of \(\{S_1,S_2,S_3\}\).
4. **Random unrelated quadruples:** not included in v1 (optional future arm).

---

## Open design choices (future)

1. Symmetrized distances \(\frac{1}{2}(d(i,j)+d(j,i))\).
2. Hop-specific vs far-specific cloze stems (v6 used these for Markov hops).
3. Unrelated random quadruples as adversarial baseline.

---

## Run

Default SLURM job uses **Gemma-2-2B** and lustre HF cache.

```bash
mkdir -p logs
sbatch src/aim3_four_point_geometry/run_experiment.slurm

# smoke (Type A only, 8 items)
sbatch --export=ALL,MAX_ITEMS=8,AIM3_TYPES=A \
  src/aim3_four_point_geometry/run_experiment.slurm

# 9B on another machine (when ready)
# sbatch --export=ALL,SAE_MODEL=google/gemma-2-9b ...
```

Outputs: `output_gemma-2-2b/layer_14/four_point_summary.json` (includes Wilcoxon +
binomial under `paired_tests`), figures for Type A four-point and Type B/C triangle slack.
