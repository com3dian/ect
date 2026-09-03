# Middle-Layer Logit Lens Evaluation of Logical Disjunction (OR)

Package: `logit_lens_aim1_OR` (sibling of `logit_lens_aim1_AND`, parallel to `output_logit_aim1_OR`).

---

## 1. Objective & Theoretical Grounding

Following the middle-layer validation of logical **conjunction (AND)** as categorical limits (pointwise minimum), this phase extends the same logit-lens protocol to logical **disjunction (OR)**.

Under the Enriched Category Theory framework, OR corresponds to **categorical colimits**, computationally defined as the **pointwise maximum** over copresheaves. The objective is to test whether the empirical middle-layer representation of \(h^{A \lor B}\) structurally aligns with the theoretical prediction \(\max(h^A, h^B)\)—mirroring what `output_logit_aim1_OR` does at the **final output logits**, but here applied at an **internal transformer layer**.

This lets us compare:
- **Where** composition is encoded (middle layer vs. final logits)
- **How** category type (especially Surreal / Mutually Exclusive) affects OR under the \(\max\) operator vs. the AND \(\min\) operator

---

## 2. Reusing Output from the Sibling AND Folder

This package does **not** re-extract \(P_A\) or \(P_B\). It **reuses the existing middle-layer AND artifacts** from the sibling folder:

**Source (read-only cache):**
```
src/logit_lens_aim1_AND/output/layer_{L}/extracted_distributions_top4096.jsonl
```

For each `(word_a, word_b)` row in that file, the OR run copies:
- `P_A`, `P_B` — token ids + probabilities (unchanged)
- pair metadata — `category_id`, `word_a`, `word_b`, `domain_seed`, etc.

and adds only the newly extracted:
- `P_A_or_B` — middle-layer logit-lens distribution for the disjunction prompt

**Requirement:** The AND logit-lens job for layer `L` must finish first. The OR layer index must match the AND cache layer (`layer_{L}`), so both runs read the same internal representation.

This mirrors how `output_logit_aim1_OR` inherits from `output_logit_aim1_AND/output/extracted_distributions_top4096.jsonl` at the final-logit level.

---

## 3. Relationship to Sibling Packages

| Package | Lens position | Operation | Extraction scope |
|---------|---------------|-----------|------------------|
| `logit_lens_aim1_AND` | Middle layer | AND (\(\land\)) | Full: `P_A`, `P_B`, `P_{A∧B}` |
| **`logit_lens_aim1_OR`** | Middle layer | OR (\(\lor\)) | **Incremental:** `P_{A∨B}` only |
| `output_logit_aim1_AND` | Final logits | AND | Full: `P_A`, `P_B`, `P_{A∧B}` |
| `output_logit_aim1_OR` | Final logits | OR | Incremental: `P_{A∨B}` only |

**Data reuse (summary):** see §2 above — `P_A` and `P_B` come from `logit_lens_aim1_AND/output/layer_{L}/`.

Only the new disjunction prompt \(P_{A \lor B}\) is forwarded through the model.

---

## 4. Model & Logit Lens Protocol

- **Model:** `Qwen/Qwen3.5-2B` (same baseline as the AND packages; override via `LOGIT_LENS_MODEL`)
- **Lens:** Classic middle-layer logit lens — residual stream after transformer block `layer_index` at the **final prompt token** (the colon `:`), optionally through final RMSNorm → `lm_head` → softmax → Top-K
- **Top-K:** `K = 4096` by default (override via `LOGIT_LENS_TOP_K`)

For each corpus pair `(word_a, word_b)` at layer `L`:

1. Load cached `P_A`, `P_B` from `logit_lens_aim1_AND` at the **same** `layer_{L}`
2. Build and forward only the disjunction prompt \(P_{A \lor B}\)
3. Extract Top-K distribution via logit lens at layer `L`
4. Compute theoretical colimit: \(\tilde{P}_{\max}(x) = \max(P_A(x), P_B(x)) / Z\)
5. Score divergence between empirical \(P_{A \lor B}\) and \(\tilde{P}_{\max}\)
6. Plot rank-aligned error curves per orthogonality category

---

## 5. Prompt Framing (Control Variables)

Identical typicality templates to the output-logit OR run:

| Prompt | Status | Template |
|--------|--------|----------|
| \(P_A\) | **Cached** (from AND logit lens) | `"Describe the typical properties and context of a {word_a}:"` |
| \(P_B\) | **Cached** (from AND logit lens) | `"Describe the typical properties and context of a {word_b}:"` |
| \(P_{A \lor B}\) | **New extraction** | `"Describe the typical properties and context of an entity that is either a {word_a} or a {word_b}:"` |

All prompts end with `:` so the lens reads the model's expectation at that final token.

---

## 6. Incremental Extraction Protocol

A lightweight, resumable extraction loop will:

1. Stream the AND layer-`L` distributions JSONL as the pair index
2. Skip pairs already present in the OR metrics file (resume support)
3. Run **one forward pass per pair** (disjunction prompt only)
4. Merge `P_A`, `P_B` (cached) + `P_A_or_B` (new) into the OR distributions JSONL
5. Flush to disk every `SAVE_EVERY` pairs (default 25)

**Prerequisite:** `logit_lens_aim1_AND` must have completed extraction for the target layer `L` before starting the OR run.

---

## 7. Computation of Categorical Colimits & Normalization

For each pair across the 5D Orthogonality Spectrum:

\[
P_{\text{theoretical\_or}}(x) = \frac{\max(P_A(x),\, P_B(x))}{\sum_{x'} \max(P_A(x'), P_B(x'))}
\]

Because the pointwise maximum does not sum to 1, re-normalization is **required** before any divergence metric.

**Primary comparison:** empirical \(P_{A \lor B}\) vs. normalized \(\max(P_A, P_B)\).

Planned metrics (aligned with `output_logit_aim1_OR`):
- L1 distance, cosine similarity
- KL divergence (both directions)
- Jensen–Shannon divergence
- Raw max mass (diagnostic: how much probability survives the max before renormalization)

---

## 8. Evaluation & Visualization

Per category (1–5 on the Orthogonality Spectrum):

1. **Aggregate metrics CSV/JSONL** — one row per pair with divergence scores
2. **Rank-aligned difference curves** — sort \(P_{A \lor B}\) tokens high→low; at each rank plot \(|\tilde{P}_{\max}(\text{token}) - P_{A \lor B}(\text{token})|\)
3. **Log-scale x-axis** — same plotting convention as AND rank-diff figures
4. **Quantile ribbons** — mean line + 5th–95th percentile across pairs

**Hypothesis of interest:** Categories 4 (Surreal) and 5 (Mutually Exclusive) may show different geometric penalty scaling under \(\max\) compared to \(\min\), and this signature may appear earlier or more sharply at certain middle layers than at the final logits.

---

## 9. Layer Selection & Output Layout

- **Default layer:** `num_hidden_layers // 2` (resolved after model load), same as AND
- **Override:** `LOGIT_LENS_LAYER=<int>`, CLI `--layer-index`, or Slurm env var

Outputs are written under **`output/layer_{L}/`** so different layers never collide:

```
logit_lens_aim1_OR/output/layer_{L}/
├── extracted_distributions_top4096.jsonl   # merged P_A, P_B (cached) + P_A_or_B
├── composition_metrics_or_max.jsonl
├── composition_metrics_or_max.csv
├── rank_diff_summary_q05_q95.npz
└── figures_rank_diff/
    ├── rank_diff_cat1.png
    ├── ...
    └── rank_diff_cat5.png
```

---

## 10. Planned Package Structure

```
logit_lens_aim1_OR/
├── README.md                 # this file
├── config.py                 # paths, model, layer, Top-K defaults
├── prompts.py                # P_{A∨B} template
├── model_io.py               # middle-layer logit lens extraction
├── metrics.py                # normalized max + divergence metrics
├── experiment.py             # incremental extraction loop + CLI
├── rank_diff.py              # rank-aligned OR error curves
├── plot_rank_diffs.py        # post-run plotting CLI
├── run_or_logit_lens.slurm   # GPU job (conda env `ect`, partition `gpu`)
├── requirements.txt
└── output/                   # gitignored run artifacts
```

Implementation will mirror `output_logit_aim1_OR` for OR-specific logic and `logit_lens_aim1_AND` for middle-layer I/O, without duplicating the AND full-extraction path.

---

## 11. Execution (planned)

**Smoke test (2 pairs, one layer):**
```bash
sbatch --export=ALL,MAX_PAIRS=2,LOGIT_LENS_LAYER=10 \
  src/logit_lens_aim1_OR/run_or_logit_lens.slurm
```

**Full run for a chosen layer:**
```bash
sbatch --export=ALL,LOGIT_LENS_LAYER=10 \
  src/logit_lens_aim1_OR/run_or_logit_lens.slurm
```

**Multi-layer sweep** (after AND caches exist for each layer):
```bash
for L in 10 12 14 16; do
  sbatch --export=ALL,LOGIT_LENS_LAYER=$L \
    src/logit_lens_aim1_OR/run_or_logit_lens.slurm
done
```

---

## 12. Implementation Defaults

1. **Cached source:** `logit_lens_aim1_AND/output/layer_{L}/extracted_distributions_top4096.jsonl`
2. **Top-K:** `K=4096` (matches AND logit-lens caches)
3. **Layer coupling:** OR run uses the same `L` as the AND cache
4. **Slurm / conda:** `ect` env, `gpu` partition, plot generation in-job (`run_or_logit_lens.slurm`)
