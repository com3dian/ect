#!/bin/bash
#SBATCH --job-name=ect_cn_aim2
#SBATCH --partition=main
#SBATCH --output=logs/ect_cn_aim2_%j.out
#SBATCH --error=logs/ect_cn_aim2_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Extract ConceptNet entailment pairs (IsA / PartOf) for Aim-2 patching.
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch src/aim2_ConceptNet_data_preparation/run_prepare_conceptnet.slurm
#
# Optional:
#   sbatch --export=ALL,CN_MAX_PAIRS=1000 \
#     src/aim2_ConceptNet_data_preparation/run_prepare_conceptnet.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PKG_DIR="${REPO_ROOT}/src/aim2_ConceptNet_data_preparation"
CONDA_ENV="${CONDA_ENV:-ect}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
DATA_DIR="${PKG_DIR}/data"
OUT_JSONL="${DATA_DIR}/conceptnet_entailment.jsonl"
OUT_COUNTS="${DATA_DIR}/conceptnet_entailment_counts.json"
ASSERTIONS="${CN_ASSERTIONS:-${DATA_DIR}/conceptnet-assertions-5.7.0.csv.gz}"
MAX_PAIRS="${CN_MAX_PAIRS:-1000}"
MIN_WEIGHT="${CN_MIN_WEIGHT:-1.0}"

mkdir -p "${REPO_ROOT}/logs" "${DATA_DIR}"

if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
else
  echo "conda.sh not found at ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  exit 1
fi

echo "Using Python: $(which python) ($(python --version))"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

echo "=== ConceptNet entailment extraction (Aim 2) ==="
echo "assertions=${ASSERTIONS}"
echo "max_pairs=${MAX_PAIRS}"

python -m aim2_ConceptNet_data_preparation.prepare_conceptnet \
  --assertions "${ASSERTIONS}" \
  --out "${OUT_JSONL}" \
  --counts "${OUT_COUNTS}" \
  --max-pairs "${MAX_PAIRS}" \
  --min-weight "${MIN_WEIGHT}"

echo "Done."
echo "Pairs:  ${OUT_JSONL}"
echo "Counts: ${OUT_COUNTS}"
