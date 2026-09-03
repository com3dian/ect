# Aim 3 v7 — Calibration (chain-rule tautology + short-token Markov)

## Why
v6 rejected Markov additivity. Before redesigning again, check:

1. **Scoring sanity (tautology)**  
   \(\log\pi(n_2 n_3 n_4\mid\mathrm{stem}(n_1))\) vs  
   \(\log\pi(n_2\mid\mathrm{stem})+\log\pi(n_3\mid\mathrm{stem}n_2)+\log\pi(n_4\mid\mathrm{stem}n_2 n_3)\)  
   → gap should be **≈ 0**. If not, fix tokenization/scoring first.

2. **Short-BPE Markov re-run**  
   Same v6 cloze claim, but only chains where each node has ≤ `MAX_BPE` pieces (default 2).  
   Natural vs shuffled_middles.

## Run

```bash
mkdir -p logs
sbatch --export=ALL,MAX_ITEMS=8 src/aim3_additive_chain_7/run_experiment.slurm
sbatch src/aim3_additive_chain_7/run_experiment.slurm
```

Outputs: `output_<model>/layer_14/…` (default `output_gemma-2-9b/`).  
Previous 2B run: `output_gemma-2-2b/`.
