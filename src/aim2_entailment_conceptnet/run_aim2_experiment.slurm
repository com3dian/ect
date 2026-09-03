#!/bin/bash
#SBATCH --job-name=ect_aim2_cn
#SBATCH --partition=gpu
#SBATCH --constraint=A100
#SBATCH --output=logs/ect_aim2_cn_%j.out
#SBATCH --error=logs/ect_aim2_cn_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

# Aim-2 ConceptNet: IsA/PartOf pairs → v_trad / v_ECT → patching sweep → figures.
#
# Submit from repo root (ConceptNet JSONL must already exist):
#   mkdir -p logs
#   sbatch src/aim2_ConceptNet_data_preparation/run_prepare_conceptnet.slurm
#
# Smoke (2 pairs, 1 layer):
#   sbatch --export=ALL,MAX_PAIRS=2,AIM2_CN_LAYERS=14,AIM2_CN_ALPHAS=0.5:1.0 \
#     src/aim2_entailment_conceptnet/run_aim2_experiment.slurm
#
# Full:
#   sbatch src/aim2_entailment_conceptnet/run_aim2_experiment.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PKG_DIR="${REPO_ROOT}/src/aim2_entailment_conceptnet"
CN_JSONL="${REPO_ROOT}/src/aim2_ConceptNet_data_preparation/data/conceptnet_entailment.jsonl"
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
export AIM2_CN_MODEL="${AIM2_CN_MODEL:-${AIM2_MODEL:-Qwen/Qwen3.5-2B}}"
export AIM2_CN_DEVICE="${AIM2_CN_DEVICE:-${AIM2_DEVICE:-cuda}}"
export AIM2_CN_DTYPE="${AIM2_CN_DTYPE:-${AIM2_DTYPE:-bfloat16}}"
export AIM2_CN_TOP_K_DIMS="${AIM2_CN_TOP_K_DIMS:-${AIM2_TOP_K_DIMS:-1000}}"
export AIM2_CN_LAYER_START="${AIM2_CN_LAYER_START:-${AIM2_LAYER_START:-10}}"
export AIM2_CN_LAYER_END="${AIM2_CN_LAYER_END:-${AIM2_LAYER_END:-20}}"
export AIM2_CN_ALPHAS="${AIM2_CN_ALPHAS:-${AIM2_ALPHAS:-0.1:0.5:1.0:2.0:5.0}}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

PAIRS_JSONL="${PAIRS_JSONL:-${CN_JSONL}}"
OUTPUT_DIR="${OUTPUT_DIR:-${PKG_DIR}/output}"

if [[ ! -f "${PAIRS_JSONL}" ]]; then
  echo "ConceptNet pairs missing: ${PAIRS_JSONL}" >&2
  echo "Run: sbatch src/aim2_ConceptNet_data_preparation/run_prepare_conceptnet.slurm" >&2
  exit 1
fi

EXP_ARGS=(
  --pairs "${PAIRS_JSONL}"
  --output-dir "${OUTPUT_DIR}"
  --model-id "${AIM2_CN_MODEL}"
  --device "${AIM2_CN_DEVICE}"
  --dtype "${AIM2_CN_DTYPE}"
  --top-k-dims "${AIM2_CN_TOP_K_DIMS}"
)

if [[ -n "${MAX_PAIRS:-}" ]]; then
  EXP_ARGS+=(--max-pairs "${MAX_PAIRS}")
fi

if [[ -n "${AIM2_CN_LAYERS:-${AIM2_LAYERS:-}}" ]]; then
  EXP_ARGS+=(--layers "${AIM2_CN_LAYERS:-${AIM2_LAYERS}}")
fi

if [[ -n "${AIM2_CN_ALPHAS:-${AIM2_ALPHAS:-}}" ]]; then
  EXP_ARGS+=(--alphas "${AIM2_CN_ALPHAS:-${AIM2_ALPHAS}}")
fi

if [[ "${EXTRACT_ONLY:-0}" == "1" ]]; then
  EXP_ARGS+=(--extract-only)
fi

if [[ "${PATCH_ONLY:-0}" == "1" ]]; then
  EXP_ARGS+=(--patch-only)
fi

echo "=== Extract + patch (ConceptNet) ==="
echo "pairs=${PAIRS_JSONL}"
python -m aim2_entailment_conceptnet.experiment "${EXP_ARGS[@]}"

echo "Done."
echo "Pairs:   ${PAIRS_JSONL}"
echo "Vectors: ${OUTPUT_DIR}/vectors/"
echo "Results: ${OUTPUT_DIR}/patching_results.csv"
echo "Figures: ${OUTPUT_DIR}/figures/"
