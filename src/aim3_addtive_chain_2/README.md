# Aim 3 v2 — Shared-trajectory additive chains

## Why this package exists
v1 (`aim3_sae_J_lens_pipeline`) scored **independent** probes
`prompt(n_i) → π(n_{i+1})`, so Σ hops vs shortcut mixed unrelated experiments
and additive gaps were systematically largely negative.

v2 builds **one shared trajectory** per item, runs **one forward**, and measures
distances between residual / SAE states at successive waypoints on that path.

## Type A prompt (redesigned)

```text
Phylogenetic semantic chain (shared trajectory):
1. {node_1}
2. {node_2}
3. {node_3}
4. {node_4}

Narrative: {context_sequence}
```

Waypoint = last token overlapping each `N. {node}` span.

## Distances
- **SAE tropical:** \(d_M(i,j)=-\ln\langle\pi_i,\pi_j\rangle\) on L1-normalized SAE
  activations at waypoints (path hops + shortcut → additive gap).
- **Logit lens:** from residual at waypoint \(i\), apply \(W_U\), score label
  tokens of node \(j\) → same gap construction.

## Run

```bash
mkdir -p logs
# smoke (Type A only)
sbatch --export=ALL,MAX_ITEMS=4,AIM3_TYPES=A \
  src/aim3_addtive_chain_2/run_experiment.slurm

# full
sbatch src/aim3_addtive_chain_2/run_experiment.slurm
```

Outputs under `output/layer_14/`.
