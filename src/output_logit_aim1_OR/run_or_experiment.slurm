#!/bin/bash
#SBATCH --job-name=ect_or_logit
#SBATCH --partition=gpu
#SBATCH --constraint=A100
#SBATCH --output=logs/ect_or_logit_%j.out
#SBATCH --error=logs/ect_or_logit_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Incremental OR extraction: inherit P_A/P_B from AND cache, extract P_{A∨B} only.
#
# Uses conda env "ect" (override with CONDA_ENV=...).
#
# Submit from repo root (gpu partition required for --gres=gpu:1):
#   mkdir -p src/output_logit_aim1_OR/logs
#   sbatch src/output_logit_aim1_OR/run_or_experiment.slurm
#
# Override partition at submit time, e.g. for AMD GPUs:
#   sbatch --partition=gpu_amd src/output_logit_aim1_OR/run_or_experiment.slurm
#
# Smoke test (2 pairs):
#   sbatch --export=ALL,MAX_PAIRS=2 src/output_logit_aim1_OR/run_or_experiment.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PKG_DIR="${REPO_ROOT}/src/output_logit_aim1_OR"
CONDA_ENV="${CONDA_ENV:-ect}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"

mkdir -p "${PKG_DIR}/logs" "${PKG_DIR}/output" "${PKG_DIR}/output/figures_rank_diff"

if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
else
  echo "conda.sh not found at ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  echo "Set CONDA_ROOT to your miniconda/anaconda install path." >&2
  exit 1
fi

echo "Using Python: $(which python) ($(python --version))"

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export LOGIT_LENS_MODEL="${LOGIT_LENS_MODEL:-Qwen/Qwen3.5-2B}"
export LOGIT_LENS_TOP_K="${LOGIT_LENS_TOP_K:-8192}"
export LOGIT_LENS_DEVICE="${LOGIT_LENS_DEVICE:-cuda}"
export LOGIT_LENS_DTYPE="${LOGIT_LENS_DTYPE:-bfloat16}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

CACHED_DISTRIBUTIONS="${CACHED_DISTRIBUTIONS:-${REPO_ROOT}/src/output_logit_aim1_AND/output/extracted_distributions_top4096.jsonl}"
DISTRIBUTIONS_JSONL="${DISTRIBUTIONS_JSONL:-${PKG_DIR}/output/extracted_distributions_top${LOGIT_LENS_TOP_K}.jsonl}"
METRICS_JSONL="${METRICS_JSONL:-${PKG_DIR}/output/composition_metrics_or_max.jsonl}"
METRICS_CSV="${METRICS_CSV:-${PKG_DIR}/output/composition_metrics_or_max.csv}"

ARGS=(
  --cached-distributions "${CACHED_DISTRIBUTIONS}"
  --distributions-jsonl "${DISTRIBUTIONS_JSONL}"
  --metrics-jsonl "${METRICS_JSONL}"
  --metrics-csv "${METRICS_CSV}"
  --model-id "${LOGIT_LENS_MODEL}"
  --top-k "${LOGIT_LENS_TOP_K}"
  --device "${LOGIT_LENS_DEVICE}"
  --dtype "${LOGIT_LENS_DTYPE}"
)

if [[ -n "${MAX_PAIRS:-}" ]]; then
  ARGS+=(--max-pairs "${MAX_PAIRS}")
fi

if [[ -n "${CATEGORIES:-}" ]]; then
  # shellcheck disable=SC2206
  CAT_ARR=(${CATEGORIES})
  ARGS+=(--categories "${CAT_ARR[@]}")
fi

python -m output_logit_aim1_OR.experiment "${ARGS[@]}"

echo "Done. Distributions: ${DISTRIBUTIONS_JSONL}"
echo "Metrics CSV: ${METRICS_CSV}"
echo "Plot rank diffs separately: sbatch src/output_logit_aim1_OR/run_or_plot_rank_diffs.slurm"
