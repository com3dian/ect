# Aim-2 SNLI data preparation

Download SNLI and keep **entailment** pairs for Aim-2 patching.

The previous Aim-2 job used five seed sentences because `load_dataset("snli")` fails on current HuggingFace `datasets` (the id must be `namespace/name`). This package loads `stanfordnlp/snli`, filters `gold_label` / `label == entailment`, then:

* if there are **more than 1000** entailment pairs → write the **first 1000**
* otherwise → write the **entire** entailment subset

Walk order is **validation → test → train**, so the first 1000 come from the official validation split.

## Run

From the repo root (conda env `ect`):

```bash
export PYTHONPATH=src
python -m aim2_SNLI_data_preperation.prepare_snli
```

CPU Slurm job (`main` partition, no GPU):

```bash
mkdir -p logs
sbatch src/aim2_SNLI_data_preperation/run_prepare_snli.slurm
```

Logs: `logs/ect_aim2_snli_<jobid>.out` / `.err`. Override the cap with `PREPARE_MAX_PAIRS` (use `0` for no cap).

## Outputs

| file | contents |
|------|----------|
| `data/snli_entailment.jsonl` | up to 1000 entailment pairs (Aim-2 JSONL schema) |
| `data/snli_entailment_counts.json` | full entailment counts per split vs how many were written |

Point Aim-2 at the JSONL with `PAIRS_JSONL=src/aim2_SNLI_data_preperation/data/snli_entailment.jsonl`.
