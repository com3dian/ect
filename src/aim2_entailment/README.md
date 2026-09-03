# Aim 2 - Probing Contextual Entailment via Activation Patching

## 1. Objective & Theoretical Grounding: Why the Jacobian?
The objective of this phase is to probe contextual entailment (IF X, THEN Y) using "internal homs" $[X, Y]$ from Enriched Category Theory (ECT). To do this in a transformer, we must move beyond static observations and use the **Jacobian Lens (J-lens)** for causal intervention.

**Why we must use the Jacobian instead of the Logit Lens:**
*   **Coordinate System Correction:** The Logit Lens multiplies $h$ by $W_U$, assuming $J_\ell = I$. The Jacobian $J_\ell$ corrects inter-layer rotation.
*   **Capturing True Causal Effects:** The J-lens asks: "If I tweak a neuron in layer $\ell$, how much will the probability of a target word change at the output?"
*   **Right Context:** $J_\ell = \partial h_L / \partial h_\ell$ links current state to future output — the ECT "Right Context".

## 2. Mathematical Interventions

### A. Traditional baseline ($v_{trad}$)
$v_{trad} = h_Y^\ell - h_X^\ell$ at layer $\ell$ (heuristic subtraction).

### B. ECT internal hom ($v_{ECT}$)
Standard J-lens pullback of the **unembedding row** of the target word (not the hypothesis hidden state):

$$v_{ECT} = J_\ell^T \cdot W_U[t], \qquad t = \text{target token}$$

$J_\ell = \partial h_L / \partial h_\ell$ is evaluated on premise $X$. Implemented via VJP:
$\; v_{ECT} = \nabla_{h_\ell}\,(h_L \cdot W_U[t])$.

Target $t$ is an explicit `target_word` if provided, otherwise the last alphabetic word of the hypothesis (e.g. "music" in "Someone is performing music.").

## 3. Execution Pipeline

1. **Data** — SNLI entailment pairs from `src/aim2_SNLI_data_preperation/data/snli_entailment.jsonl` (clean premise, corrupted baseline, hypothesis $Y$).
2. **Vector extraction** — $v_{trad}$, $v_{ECT}$; then **match magnitudes**: $v_{ECT} \leftarrow (v_{ECT}/\|v_{ECT}\|_2)\,\|v_{trad}\|_2$ so $\alpha$ is the same physical force. Top-1000 dim slice → `output/vectors/{pair_id}_L{L}.safetensors`.
3. **Activation patching** — `register_forward_hook` on layer $\ell$: $h \leftarrow h + \alpha v$. The hook **returns** the patched residual. Intervention is applied on the **corrupted** prompt, from the last prompt token through the hypothesis tokens (so scored log-probs actually see the patch).
4. **Sweep** — layers 10–20, $\alpha \in \{0.1, 0.5, 1, 2, 5\}$.
5. **Measurement** — mean hypothesis-token log-prob; A/B CSV: `output/patching_results.csv`.
6. **Figures** — clean vs corrupted vs patched log-prob, trad vs ECT: `output/figures/`.

## 4. Package layout

```
src/aim2_entailment/
  experiment.py        # extract + patch sweep + figures
  plot_patching.py     # clean / corrupted / patched × trad / ECT
  jacobian.py          # v_ECT VJP + Top-K slicing
  vectors.py           # safetensors I/O
  patching.py          # forward hooks
  run_aim2_experiment.slurm
  run_aim2_plot.slurm
```

## 5. Run

```bash
mkdir -p logs

# SNLI entailment JSONL (once)
sbatch src/aim2_SNLI_data_preperation/run_prepare_snli.slurm

# Smoke (2 SNLI pairs, 1 layer)
sbatch --export=ALL,MAX_PAIRS=2,AIM2_LAYERS=14,AIM2_ALPHAS=0.5:1.0 \
  src/aim2_entailment/run_aim2_experiment.slurm

# Full (all SNLI pairs in the JSONL + layer/alpha sweep)
sbatch src/aim2_entailment/run_aim2_experiment.slurm

# Replot existing CSV (CPU)
sbatch src/aim2_entailment/run_aim2_plot.slurm

# Per-sample layer-12: baselines + v_trad + v_ECT; sorted ECT vs trad diff (CPU)
sbatch src/aim2_entailment/run_aim2_plot_layer12_samples.slurm
```

**Env overrides:** `AIM2_MODEL` (default `Qwen/Qwen3.5-2B`), `AIM2_TOP_K_DIMS`, `AIM2_LAYER_START/END`, `AIM2_ALPHAS`, `MAX_PAIRS`, `PAIRS_JSONL`.

**Outputs:**
* pairs: `src/aim2_SNLI_data_preperation/data/snli_entailment.jsonl`
* `output/vectors/*.safetensors`
* `output/patching_results.csv` / `.jsonl`
* `output/figures/patching_logprob_trad_vs_ect.png`
* `output/figures/patching_trad_vs_ect_overlay.png`
