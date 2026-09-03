#!/bin/bash
#SBATCH --job-name=ect_aim2_snli
#SBATCH --partition=main
#SBATCH --output=logs/ect_aim2_snli_%j.out
#SBATCH --error=logs/ect_aim2_snli_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Download SNLI, keep entailment pairs, cap at 1000.
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch src/aim2_SNLI_data_preperation/run_prepare_snli.slurm
#
# Optional:
#   sbatch --export=ALL,PREPARE_MAX_PAIRS=1000 \
#     src/aim2_SNLI_data_preperation/run_prepare_snli.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PKG_DIR="${REPO_ROOT}/src/aim2_SNLI_data_preperation"
CONDA_ENV="${CONDA_ENV:-ect}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
OUT_JSONL="${PKG_DIR}/data/snli_entailment.jsonl"
OUT_COUNTS="${PKG_DIR}/data/snli_entailment_counts.json"
PREPARE_MAX="${PREPARE_MAX_PAIRS:-1000}"

mkdir -p "${REPO_ROOT}/logs" "${PKG_DIR}/data"

if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
else
  echo "conda.sh not found at ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  exit 1
fi

echo "Using Python: $(which python) ($(python --version))"
echo "max_pairs=${PREPARE_MAX}"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

echo "=== Prepare SNLI entailment pairs ==="
python -m aim2_SNLI_data_preperation.prepare_snli \
  --out "${OUT_JSONL}" \
  --counts "${OUT_COUNTS}" \
  --max-pairs "${PREPARE_MAX}"

echo "Done."
echo "Pairs:  ${OUT_JSONL}"
echo "Counts: ${OUT_COUNTS}"
