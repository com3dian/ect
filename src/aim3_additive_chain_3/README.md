# Aim 3 v3 — Shared-trajectory additive chains in **J-lens** space

## vs previous packages
| Package | Distance space |
|---|---|
| `aim3_sae_J_lens_pipeline` | Independent probes + token \(d_M\) / SAE |
| `aim3_addtive_chain_2` | Shared trajectory + **SAE** tropical |
| **`aim3_additive_chain_3`** | Shared trajectory + **J-lens** tropical |

## Method
1. Same shared Type A prompt as v2 (one forward through `1. n1 … 4. n4`).
2. Residual \(h_{14}\) at each node waypoint.
3. Truncated Jacobian \(J\) at the **last** waypoint (Top-K vocab × Top-K residual).
4. Project each waypoint into J-space: \(z_i = J\, h_i[\mathrm{resid}]\).
5. Tropical distance on softplus / L1-normalized \(z\):  
   \(d_M(i,j)=-\ln\langle\pi_i,\pi_j\rangle\).
6. Additive gap: \(d(n_1,n_4)-\sum\) hops — same plot layout as v2.

## Run

```bash
mkdir -p logs
sbatch --export=ALL,MAX_ITEMS=4,AIM3_TYPES=A \
  src/aim3_additive_chain_3/run_experiment.slurm

sbatch src/aim3_additive_chain_3/run_experiment.slurm
```

Figure: `output/layer_14/figures/type_a_j_lens_trajectory.png`
