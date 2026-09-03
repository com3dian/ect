# Aim 3 v6 — Markov cloze factorization with **true sequence** \(d_M=-\ln\pi\)

## What changed vs v4 / v5

| Issue before | v6 fix |
|---|---|
| Fingerprint TVD / feature overlap | Dropped — not \(d_M\) |
| Shared list trajectory + label jump | **Markov cloze**: only the conditioning concept in the prefix |
| Mean subword logprob | **Sum** of token logprobs \(=\log\pi(\text{string}\mid\text{prefix})\) |
| No controls | **shuffled_middles** and **reversed** chains |
| Only absolute gap | Also **relative gap** \((D-L)/L\) |

## Claim (Type A)

\[
\pi(n_4\mid n_1)\;\stackrel{?}{\approx}\;
\pi(n_2\mid n_1)\,\pi(n_3\mid n_2)\,\pi(n_4\mid n_3)
\]

Equivalently with \(d_M=-\sum_i\log\pi(t_i\mid\text{prefix},t_{<i})\):

\[
d(n_1,n_4)\;\stackrel{?}{\approx}\;
d(n_1,n_2)+d(n_2,n_3)+d(n_3,n_4)
\]

Cloze stem (hop):

```text
Taxonomic refinement in English.
Concept: "{n_i}"
A more specific kind of that concept is
```

Shortcut uses “much more specific instance” after `{n_1}`.

## Run

```bash
mkdir -p logs
sbatch --export=ALL,MAX_ITEMS=4,AIM3_TYPES=A \
  src/aim3_additive_chain_6/run_experiment.slurm

sbatch src/aim3_additive_chain_6/run_experiment.slurm
```

Outputs under `output_<model>/layer_14/` (default model `google/gemma-2-9b` → `output_gemma-2-9b/`).  
Previous Gemma-2-2B run preserved as `output_gemma-2-2b/`.
