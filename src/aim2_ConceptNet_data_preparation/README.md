# ConceptNet Entailment Extraction for Aim 2

Extract **entailment-only** pairs from ConceptNet 5.7 for the Aim-2 activation patching pipeline (`aim2_entailment`).

## Objective

Build a controlled entailment dataset from ConceptNet strict hierarchical edges, as an alternative/complement to SNLI. Only **Category 1–style strict entailment** is used — no composition, pathfinding, or paradox pairs.

## ConceptNet → entailment mapping

| Edge | ConceptNet example | Premise (X) | Hypothesis (Y) |
|------|-------------------|-------------|----------------|
| **IsA** | `[dog] → [animal]` | This is a dog. | This is an animal. |
| **PartOf** | `[wheel] → [car]` | This is a car. | This has a wheel. |

Non-English nodes and low-weight edges are filtered out.

## Output

JSONL compatible with `aim2_entailment` / SNLI schema:

```json
{
  "pair_id": "conceptnet_isa_0",
  "premise": "This is a dog.",
  "hypothesis": "This is an animal.",
  "target_word": "animal",
  "label": "entailment",
  "source": "conceptnet",
  "conceptnet_edge": "IsA",
  "conceptnet_start": "dog",
  "conceptnet_end": "animal",
  "prompts": {
    "premise": "This is a dog.",
    "corrupted": "It is the case that:",
    "hypothesis": "This is an animal."
  }
}
```

**Files:**
* `data/conceptnet_entailment.jsonl`
* `data/conceptnet_entailment_counts.json`
* `data/conceptnet-assertions-5.7.0.csv.gz` (cached download)

## Run

```bash
mkdir -p logs

# Default: up to 1000 entailment pairs
sbatch src/aim2_ConceptNet_data_preparation/run_prepare_conceptnet.slurm

# Custom cap
sbatch --export=ALL,CN_MAX_PAIRS=500 \
  src/aim2_ConceptNet_data_preparation/run_prepare_conceptnet.slurm
```

**Local (CPU):**

```bash
conda activate ect
export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
python -m aim2_ConceptNet_data_preparation.prepare_conceptnet --max-pairs 1000
```

**Use with Aim-2 experiment:**

```bash
sbatch src/aim2_entailment_conceptnet/run_aim2_experiment.slurm
```

See [`aim2_entailment_conceptnet`](../aim2_entailment_conceptnet/README.md) for the full patching pipeline.

**Env overrides:** `CN_ASSERTIONS`, `CN_MAX_PAIRS`, `CN_MIN_WEIGHT`.
