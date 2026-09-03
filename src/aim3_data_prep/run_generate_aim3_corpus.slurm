#!/bin/bash
#SBATCH --job-name=ect_aim3_corpus
#SBATCH --partition=main
#SBATCH --output=logs/ect_aim3_corpus_%j.out
#SBATCH --error=logs/ect_aim3_corpus_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

# Generate Aim-3 tropical geometry corpus via SURF AI Hub / WiLLMa.
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch src/aim3_data_prep/run_generate_aim3_corpus.slurm
#
# Optional overrides:
#   sbatch --export=ALL,AIM3_BATCHES=4,AIM3_TYPES=A,B,C \
#     src/aim3_data_prep/run_generate_aim3_corpus.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PKG_DIR="${REPO_ROOT}/src/aim3_data_prep"
AIM1_ENV="${REPO_ROOT}/src/vocabulary_gen_aim1_AND/.env"
CONDA_ENV="${CONDA_ENV:-ect}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"

OUT_JSON="${AIM3_OUT_JSON:-${PKG_DIR}/data/tropical_geometry_corpus.json}"
OUT_JSONL="${AIM3_OUT_JSONL:-${PKG_DIR}/data/tropical_geometry_corpus.jsonl}"
TYPES="${AIM3_TYPES:-A,B,C}"
ITEMS_PER_BATCH="${AIM3_ITEMS_PER_BATCH:-50}"
BATCHES="${AIM3_BATCHES:-3}"
TARGET="${AIM3_TARGET_PER_TYPE:-150}"
CONCURRENCY="${AIM3_CONCURRENCY:-1}"

mkdir -p "${REPO_ROOT}/logs" "${PKG_DIR}/data"

if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
else
  echo "conda.sh not found at ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  exit 1
fi

# Reuse Aim-1 WiLLMa / SURF credentials (WILLMA_API_KEY, optional BASE_URL / MODEL).
if [[ -f "${AIM1_ENV}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${AIM1_ENV}"
  set +a
  echo "Loaded WiLLMa credentials from ${AIM1_ENV}"
else
  echo "WARNING: ${AIM1_ENV} not found; relying on exported WILLMA_API_KEY" >&2
fi

if [[ -z "${WILLMA_API_KEY:-${OPENAI_API_KEY:-}}" ]]; then
  echo "Missing WILLMA_API_KEY / OPENAI_API_KEY" >&2
  exit 1
fi

echo "Using Python: $(which python) ($(python --version))"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Keep Qwen from spending the budget on hidden thinking traces.
export WILLMA_DISABLE_THINKING="${WILLMA_DISABLE_THINKING:-1}"

echo "=== Aim-3 tropical geometry corpus generation ==="
echo "types=${TYPES}"
echo "items_per_batch=${ITEMS_PER_BATCH}"
echo "batches_per_type=${BATCHES}"
echo "target_per_type=${TARGET}"
echo "model=${WILLMA_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}"
echo "base_url=${WILLMA_BASE_URL:-https://api.willma.surf.nl/v0}"

python -m aim3_data_prep.generate_aim3_corpus \
  --types "${TYPES}" \
  --items-per-batch "${ITEMS_PER_BATCH}" \
  --batches-per-type "${BATCHES}" \
  --target-per-type "${TARGET}" \
  --concurrency "${CONCURRENCY}" \
  --output-json "${OUT_JSON}" \
  --output-jsonl "${OUT_JSONL}"

echo "Done."
echo "JSON:  ${OUT_JSON}"
echo "JSONL: ${OUT_JSONL}"
