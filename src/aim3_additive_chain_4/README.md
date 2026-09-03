# Aim 3 v4 — Shared trajectory + **true** \(d_M=-\ln\pi(y\mid x)\)

## Point of this package
v2/v3 used \(-\ln\langle\pi_i,\pi_j\rangle\) on SAE/J **features** (not LM \(\pi\)).
v4 keeps the ECT claim: distances are real model conditionals.

## Distance definition (Type A)
Shared prompt:

```text
Phylogenetic semantic chain (shared trajectory):
1. {n1}
2. {n2}
3. {n3}
4. {n4}
…
```

- Prefix through \(n_i\) = text up to and including that node line.
- Hop \(n_i\to n_{i+1}\): \(d=-\ln\pi(n_{i+1}\mid \text{prefix through }n_i)\)
- Shortcut \(n_1\to n_4\): \(d=-\ln\pi(n_4\mid \text{prefix through }n_1)\)

Shortcut scores the **label jump** (not the whole middle text), so this is
**not** a chain-rule tautology.

## Run

```bash
mkdir -p logs
sbatch --export=ALL,MAX_ITEMS=4,AIM3_TYPES=A \
  src/aim3_additive_chain_4/run_experiment.slurm

sbatch src/aim3_additive_chain_4/run_experiment.slurm
```

Figure: `output/layer_14/figures/type_a_true_ln_pi.png`
