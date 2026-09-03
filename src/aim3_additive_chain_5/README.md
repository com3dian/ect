# Aim 3 v5 — Shared trajectory + **TVD / half-L1 path costs**

## What this measures
At each node waypoint, take a fingerprint \(\pi\) (L1-normalized SAE features,
and optionally J-lens softplus features).

Step cost between two fingerprints:
\[
\delta(a,b)=\tfrac12\sum_i |\pi_a_i-\pi_b_i|
=\mathrm{TVD}(\pi_a,\pi_b)
\]

- Path: \(L=\delta(n_1,n_2)+\delta(n_2,n_3)+\delta(n_3,n_4)\)
- Direct: \(D=\delta(n_1,n_4)\)
- Gap: \(D-L\) (always \(\le 0\) for TVD; **near 0 ⇒ additive / geodesic path**)

This is the “add up how different the fingerprints are, feature by feature,
then take half” metric.

## Run

```bash
mkdir -p logs
sbatch --export=ALL,MAX_ITEMS=4,AIM3_TYPES=A \
  src/aim3_additive_chain_5/run_experiment.slurm

# SAE only (skip Jacobian, faster):
sbatch --export=ALL,NO_J=1 \
  src/aim3_additive_chain_5/run_experiment.slurm

sbatch src/aim3_additive_chain_5/run_experiment.slurm
```

Figures:
- `output/layer_14/figures/type_a_tvd_path.png` — SAE path costs + SAE/J gap overlay
- `output/layer_14/figures/type_a_tvd_path_j_lens.png` — J-lens path costs + zoomed gap hist
