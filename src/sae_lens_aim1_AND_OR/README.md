# Phase 3: Empirical Evaluation of Categorical Limits and Colimits via Sparse Autoencoder (SAE) Lens

## 1. Project Objective & Theoretical Grounding
This pipeline aims to validate the hypothesis derived from Enriched Category Theory (ECT): Large Language Models (LLMs) encode logical composition in their latent space using categorical limits and colimits.
*   **Logical Conjunction (AND):** Modeled as a weighted limit, corresponding to the **pointwise minimum** ($\min(h^A, h^B)$) over copresheaves.
*   **Logical Disjunction (OR):** Modeled as a categorical colimit, corresponding to the **pointwise maximum** ($\max(h^A, h^B)$).

Previous experiments utilizing Logit Lens and Raw Cosine Distance suffered from "normalization explosion" and dense space "catastrophic blending". This script shifts the observation space to the highly sparse, non-negative topological basis provided by Monosemantic Features (extracted via SAEs).

## 2. Infrastructure & Model Selection
*   **Target Model:** `google/gemma-2-2b` (dense model suitable for single-GPU analysis; required for Gemma Scope SAEs).
*   **SAE Framework:** `SAELens` (Apollo Research).
*   **SAE Dictionary:** Gemma Scope residual stream, release `gemma-scope-2b-pt-res-canonical`.
*   **Default layer:** 14 (`layer_14/width_16k/canonical`). Override with `SAE_LAYER` / `--layer-index`.

## 3. Data Pipeline & Prompt Framing
We reuse the generated "5D Orthogonality Spectrum" corpus from `vocabulary_gen_aim1_AND`. For each concept pair (`word_a`, `word_b`), four prompts are built with the **same typicality templates** as `output_logit_aim1_AND` / `output_logit_aim1_OR` (final colon `:` is the extraction position):

1.  **$P_A$ (Concept A):** `"Describe the typical properties and context of a {word_a}:"`
2.  **$P_B$ (Concept B):** `"Describe the typical properties and context of a {word_b}:"`
3.  **$P_{AB\_AND}$ (Conjunction):** `"Describe the typical properties and context of an entity that is both a {word_a} and a {word_b}:"`
4.  **$P_{AB\_OR}$ (Disjunction):** `"Describe the typical properties and context of an entity that is either a {word_a} or a {word_b}:"`

## 4. SAE Extraction & Mathematical Baselines
For each prompt, extract the residual stream $h$ at the final token and pass it through the SAE to obtain a sparse feature activation vector $V$ (e.g. 16{,}384 dims for width-16k).
All vectors ($V_A$, $V_B$, $V_{AND}$, $V_{OR}$) are **$L_1$-normalized** to sum to 1.0 (semantic energy distributions).

Theoretical baselines:
*   **Categorical Limit (Min):** $V_{min} = \mathrm{Norm}(\min(V_A, V_B))$
*   **Categorical Colimit (Max):** $V_{max} = \mathrm{Norm}(\max(V_A, V_B))$
*   **Arithmetic Mean:** $V_{mean} = \mathrm{Norm}((V_A + V_B) / 2)$
*   **Product:** $V_{prod} = \mathrm{Norm}(V_A \times V_B)$

## 5. Metrics: Topological Geometry in Sparse Space
1.  **Total Variation Distance (TVD / half-L1):**
    *   $TVD_{AND} = 0.5 \times \sum |V_{AND} - V_{theory}|$ for theory $\in \{\min, \max, \mathrm{mean}, \mathrm{prod}\}$
    *   $TVD_{OR} = 0.5 \times \sum |V_{OR} - V_{theory}|$ similarly
2.  **Jaccard Similarity** over active feature indices ($F_A$, $F_B$):
    $J(A, B) = |F_A \cap F_B| / |F_A \cup F_B|$

## 6. Outputs & Execution
Per-layer outputs under `output/layer_{L}/`:
*   `sae_composition_metrics.jsonl` / `.csv` — per-pair TVD + Jaccard (resumable)
*   `category_summary.csv` — mean metrics grouped by category
*   optional sparse feature dumps if enabled

```bash
mkdir -p logs
# Smoke test
sbatch --export=ALL,MAX_PAIRS=2,SAE_LAYER=14 \
  src/sae_lens_aim1_AND_OR/run_sae_experiment.slurm

# Full run (layer 14)
sbatch --export=ALL,SAE_LAYER=14 \
  src/sae_lens_aim1_AND_OR/run_sae_experiment.slurm
```
