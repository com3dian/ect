# Aim 2 — ConceptNet Entailment via Activation Patching

Same pipeline as [`aim2_entailment`](../aim2_entailment/README.md), but uses **ConceptNet IsA / PartOf** entailment pairs instead of SNLI.

## Data

Pairs from `aim2_ConceptNet_data_preparation`:

```bash
sbatch src/aim2_ConceptNet_data_preparation/run_prepare_conceptnet.slurm
```

→ `src/aim2_ConceptNet_data_preparation/data/conceptnet_entailment.jsonl`

Example pair:
- **premise:** `This is a dog.`
- **hypothesis:** `This is an animal.`
- **corrupted:** `It is the case that:`

## Pipeline

1. **Data** — ConceptNet entailment JSONL
2. **Vector extraction** — $v_{trad}$, $v_{ECT}$ (Top-K safetensors)
3. **Activation patching** — sweep layers 10–20, $\alpha \in \{0.1, 0.5, 1, 2, 5\}$
4. **Measurement** — mean hypothesis-token log-prob → CSV
5. **Figures** — trad vs ECT, per-sample layer plots

## Package layout

```
src/aim2_entailment_conceptnet/
  experiment.py
  plot_patching.py
  plot_layer_samples.py
  jacobian.py
  vectors.py
  patching.py
  run_aim2_experiment.slurm
  run_aim2_plot.slurm
  run_aim2_plot_layer12_samples.slurm
```

## Run

```bash
mkdir -p logs

# 1) Prepare ConceptNet pairs (once)
sbatch src/aim2_ConceptNet_data_preparation/run_prepare_conceptnet.slurm

# 2) Smoke (2 pairs, 1 layer)
sbatch --export=ALL,MAX_PAIRS=2,AIM2_CN_LAYERS=14,AIM2_CN_ALPHAS=0.5:1.0 \
  src/aim2_entailment_conceptnet/run_aim2_experiment.slurm

# 3) Full experiment
sbatch src/aim2_entailment_conceptnet/run_aim2_experiment.slurm

# 4) Replot CSV (CPU)
sbatch src/aim2_entailment_conceptnet/run_aim2_plot.slurm

# 5) Per-sample layer-12 plots (CPU)
sbatch src/aim2_entailment_conceptnet/run_aim2_plot_layer12_samples.slurm
```

**Env overrides:** `AIM2_CN_MODEL`, `AIM2_CN_TOP_K_DIMS`, `AIM2_CN_LAYER_START/END`, `AIM2_CN_ALPHAS`, `MAX_PAIRS`, `PAIRS_JSONL`.

(Falls back to `AIM2_*` env vars if `AIM2_CN_*` is unset.)

**Outputs:**
* `output/vectors/*.safetensors`
* `output/patching_results.csv` / `.jsonl`
* `output/figures/`
