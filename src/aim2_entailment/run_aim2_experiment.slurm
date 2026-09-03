#!/bin/bash
#SBATCH --job-name=ect_aim2
#SBATCH --partition=gpu
#SBATCH --constraint=A100
#SBATCH --output=logs/ect_aim2_%j.out
#SBATCH --error=logs/ect_aim2_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

# Aim-2: SNLI entailment pairs → v_trad / v_ECT extraction → patching sweep → figures.
#
# Submit from repo root (SNLI JSONL must already exist):
#   mkdir -p logs
#   sbatch src/aim2_SNLI_data_preperation/run_prepare_snli.slurm
#
# Smoke (2 SNLI pairs, 1 layer). Use ':' not ',' in AIM2_ALPHAS —
# sbatch --export splits on commas.
#   sbatch --export=ALL,MAX_PAIRS=2,AIM2_LAYERS=14,AIM2_ALPHAS=0.5:1.0 \
#     src/aim2_entailment/run_aim2_experiment.slurm
#
# Full (all SNLI pairs in the JSONL, layers 10–20, all alphas):
#   sbatch src/aim2_entailment/run_aim2_experiment.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PKG_DIR="${REPO_ROOT}/src/aim2_entailment"
SNLI_JSONL="${REPO_ROOT}/src/aim2_SNLI_data_preperation/data/snli_entailment.jsonl"
CONDA_ENV="${CONDA_ENV:-ect}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"

mkdir -p "${REPO_ROOT}/logs" "${PKG_DIR}/output"

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
export AIM2_MODEL="${AIM2_MODEL:-Qwen/Qwen3.5-2B}"
export AIM2_DEVICE="${AIM2_DEVICE:-cuda}"
export AIM2_DTYPE="${AIM2_DTYPE:-bfloat16}"
export AIM2_TOP_K_DIMS="${AIM2_TOP_K_DIMS:-1000}"
export AIM2_LAYER_START="${AIM2_LAYER_START:-10}"
export AIM2_LAYER_END="${AIM2_LAYER_END:-20}"
export AIM2_ALPHAS="${AIM2_ALPHAS:-0.1:0.5:1.0:2.0:5.0}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

PAIRS_JSONL="${PAIRS_JSONL:-${SNLI_JSONL}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PKG_DIR}/output}"

if [[ ! -f "${PAIRS_JSONL}" ]]; then
  echo "SNLI pairs missing: ${PAIRS_JSONL}" >&2
  echo "Run: sbatch src/aim2_SNLI_data_preperation/run_prepare_snli.slurm" >&2
  exit 1
fi

EXP_ARGS=(
  --pairs "${PAIRS_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --model-id "${AIM2_MODEL}"
  --device "${AIM2_DEVICE}"
  --dtype "${AIM2_DTYPE}"
  --top-k-dims "${AIM2_TOP_K_DIMS}"
)

if [[ -n "${MAX_PAIRS:-}" ]]; then
  EXP_ARGS+=(--max-pairs "${MAX_PAIRS}")
fi

if [[ -n "${AIM2_LAYERS:-}" ]]; then
  EXP_ARGS+=(--layers "${AIM2_LAYERS}")
fi

if [[ -n "${AIM2_ALPHAS:-}" ]]; then
  EXP_ARGS+=(--alphas "${AIM2_ALPHAS}")
fi

if [[ "${EXTRACT_ONLY:-0}" == "1" ]]; then
  EXP_ARGS+=(--extract-only)
fi

if [[ "${PATCH_ONLY:-0}" == "1" ]]; then
  EXP_ARGS+=(--patch-only)
fi

echo "=== Extract + patch (SNLI) ==="
echo "pairs=${PAIRS_JSONL}"
python -m aim2_entailment.experiment "${EXP_ARGS[@]}"

echo "Done."
echo "Pairs:   ${PAIRS_JSONL}"
echo "Vectors: ${OUTPUT_DIR}/vectors/"
echo "Results: ${OUTPUT_DIR}/patching_results.csv"
echo "Figures: ${OUTPUT_DIR}/figures/"
