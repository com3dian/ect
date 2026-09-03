# Differential SAE Lens via Syntactic Baseline Subtraction

## 1. Problem Diagnosis & Objective
In standard SAE feature extraction, raw prompt activations are dominated by shared template tokens (syntax, colons, formatting). Consequently $\min$, $\mathrm{mean}$, and $\max$ of $V_A$ and $V_B$ collapse.

**Objective:** **Differential Baseline Subtraction** $\tilde{V} = \mathrm{ReLU}(V - V_{\mathrm{base}})$ to cancel syntactic noise before ECT limit / colimit comparisons.

---

## 2. Infrastructure
* **Model:** `google/gemma-2-2b`
* **SAE:** Gemma Scope residual via `sae_lens` (default `gemma-scope-2b-pt-res-canonical`, `layer_14/width_16k/canonical`)
* **Optional:** `gemma-scope-2b-pt-res` + `average_l0_71` / `width_65k` via env
* **Device:** `bfloat16` on one CUDA GPU (A100)

---

## 3. Prompts

```python
TEMPLATE_A / B / AND / OR   # concept prompts (same typicality templates)
TEMPLATE_BASE_SINGLE = "Describe the typical properties and context of an entity:"
TEMPLATE_BASE_COMPOSED = "Describe the typical properties and context of an entity that is:"
```

Baselines are pair-independent and extracted once per run.

---

## 4. Differential Feature Extraction Protocol

**Forward pass & SAE:** residual $h$ at the final token (`:`), SAE encode →
$V_A, V_B, V_{AND}, V_{OR}, V_{\mathrm{base\_single}}, V_{\mathrm{base\_comp}} \ge 0$.

**ReLU denoising:**
$$
\begin{aligned}
\tilde{V}_A &= \mathrm{ReLU}(V_A - V_{\mathrm{base\_single}}) \\
\tilde{V}_B &= \mathrm{ReLU}(V_B - V_{\mathrm{base\_single}}) \\
\tilde{V}_{AND} &= \mathrm{ReLU}(V_{AND} - V_{\mathrm{base\_comp}}) \\
\tilde{V}_{OR} &= \mathrm{ReLU}(V_{OR} - V_{\mathrm{base\_comp}})
\end{aligned}
$$

**$L_1$ energy normalization** ($\epsilon = 10^{-12}$):
$$\hat{V} = \tilde{V} / (\|\tilde{V}\|_1 + \epsilon)$$

---

## 5. Theoretical Baselines & Metrics

**Theories from denoised $\tilde{V}$ (not from $\hat{V}$):**
$$
\begin{aligned}
V_{\mathrm{theo\_min}} &= \min(\tilde{V}_A,\tilde{V}_B) / \|\cdots\|_1 \\
V_{\mathrm{theo\_max}} &= \max(\tilde{V}_A,\tilde{V}_B) / \|\cdots\|_1 \\
V_{\mathrm{theo\_mean}} &= (\tilde{V}_A+\tilde{V}_B) / \|\tilde{V}_A+\tilde{V}_B\|_1 \\
V_{\mathrm{theo\_prod}} &= (\tilde{V}_A \odot \tilde{V}_B) / \|\cdots\|_1
\end{aligned}
$$

**TVD:** $\mathrm{TVD}(\hat{V}_{\mathrm{AND/OR}}, V_{\mathrm{theo\_*}}) = \tfrac12 \sum_i |\cdot|$

Also logged:
* $L_0$ **before** and **after** subtraction
* per-pair operator winners + category **win-rates**
* Jaccard($A$,$B$) on active tilde features

---

## 6. Plots
1. **TVD box/bar** → `figures_tvd/`
2. **Rank-diff ribbons** (AND & OR) → `figures_rank_diff/{and,or}/`

---

## 7. Outputs & run

Under `output/layer_{L}/`:
* `results_sae_subtraction.jsonl` / `.csv` / `.json`
* `category_summary.csv` (mean TVD + win-rates)
* `sae_feature_distributions.jsonl`
* figure dirs above

```bash
mkdir -p logs

# Smoke
sbatch --export=ALL,MAX_PAIRS=2,SAE_LAYER=14 \
  src/sae_lens_aim1_AND_OR_baseline_substraction/run_sae_experiment.slurm

# Full (metrics + both plot types)
sbatch --export=ALL,SAE_LAYER=14 \
  src/sae_lens_aim1_AND_OR_baseline_substraction/run_sae_experiment.slurm
```
