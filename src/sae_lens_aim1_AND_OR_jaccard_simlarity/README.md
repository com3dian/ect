# Phase 3.2: Discrete Set Theory & Jaccard Similarity in SAE Space

## 1. Problem Diagnosis & Shift to Set Theory
Continuous distance metrics (e.g. TVD after $L_1$ normalization) can fail in sparse SAE space: when concepts are orthogonal, $\min(V_A,V_B)$ approaches zero and normalizing amplifies tail noise.

**Objective:** Evaluate ECT limits (AND) and colimits (OR) with **discrete set theory** and **Jaccard similarity** on strongly active feature indices — no normalization / TVD.

## 2. Core Metric
Treat active SAE features as discrete concept-neuron sets. Jaccard $= |A \cap B| / |A \cup B|$.

* Orthogonal concepts → empty intersection → Jaccard $= 0$.
* Directly tests whether the model’s AND behaves like intersection and OR like union.

## 3. Pipeline

**Step 1 — Binarization**  
Extract raw SAE activations for $P_A$, $P_B$, $P_{AND}$, $P_{OR}$ at the final token (`:`).  
Threshold at $\theta = 1.0$ (override with `SAE_JACCARD_THRESHOLD` / `--threshold`) → sets $F_A$, $F_B$, $F_{AND}$, $F_{OR}$.

**Step 2 — Theory sets**
* Limit (AND / min): $F_{\mathrm{theo\_min}} = F_A \cap F_B$
* Colimit (OR / max): $F_{\mathrm{theo\_max}} = F_A \cup F_B$

**Step 3 — Jaccard**
* AND: $J(F_{AND}, F_{\mathrm{theo\_min}})$
* OR: $J(F_{OR}, F_{\mathrm{theo\_max}})$

**Step 4 — Control**
* AND vs union: $J(F_{AND}, F_{\mathrm{theo\_max}})$ — if AND is a true limit, ∩-Jaccard should beat ∪-Jaccard.

## 4. Outputs
Under `output/layer_{L}/`:
* `results_sae_jaccard.jsonl` / `.csv` / `.json`
* `category_summary.csv` (mean Jaccard + rates where AND prefers ∩ / OR prefers ∪)
* `figures_jaccard/` — per-category boxes, AND ∩-vs-∪ focus plot, summary bars

## 5. Run

```bash
mkdir -p logs

# Smoke
sbatch --export=ALL,MAX_PAIRS=2,SAE_LAYER=14 \
  src/sae_lens_aim1_AND_OR_jaccard_simlarity/run_sae_experiment.slurm

# Full (metrics + plots)
sbatch --export=ALL,SAE_LAYER=14 \
  src/sae_lens_aim1_AND_OR_jaccard_simlarity/run_sae_experiment.slurm

# CPU replot only
sbatch --export=ALL,SAE_LAYER=14 \
  src/sae_lens_aim1_AND_OR_jaccard_simlarity/run_sae_plot_jaccard.slurm
```
