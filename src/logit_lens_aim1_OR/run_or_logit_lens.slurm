#!/bin/bash
#SBATCH --job-name=ect_or_lens
#SBATCH --partition=gpu
#SBATCH --constraint=A100
#SBATCH --output=logs/ect_or_lens_%j.out
#SBATCH --error=logs/ect_or_lens_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G

# Incremental OR middle-layer extraction:
# inherit P_A/P_B from logit_lens_aim1_AND/output/layer_{L}/, extract P_{A∨B} only.
#
# Uses conda env "ect" (override with CONDA_ENV=...).
#
# Submit from repo root (gpu partition required for --gres=gpu:1):
#   mkdir -p src/logit_lens_aim1_OR/logs
#   sbatch --export=ALL,LOGIT_LENS_LAYER=10 src/logit_lens_aim1_OR/run_or_logit_lens.slurm
#
# Smoke test (2 pairs):
#   sbatch --export=ALL,MAX_PAIRS=2,LOGIT_LENS_LAYER=10 \
#     src/logit_lens_aim1_OR/run_or_logit_lens.slurm
#
# Multi-layer sweep (after AND caches exist):
#   for L in 10 12 14 16; do
#     sbatch --export=ALL,LOGIT_LENS_LAYER=$L src/logit_lens_aim1_OR/run_or_logit_lens.slurm
#   done

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PKG_DIR="${REPO_ROOT}/src/logit_lens_aim1_OR"
CONDA_ENV="${CONDA_ENV:-ect}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"

mkdir -p "${PKG_DIR}/logs" "${PKG_DIR}/output"

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
export LOGIT_LENS_TOP_K="${LOGIT_LENS_TOP_K:-4096}"
export LOGIT_LENS_DEVICE="${LOGIT_LENS_DEVICE:-cuda}"
export LOGIT_LENS_DTYPE="${LOGIT_LENS_DTYPE:-bfloat16}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

# Default middle-ish layer used in AND runs; override with LOGIT_LENS_LAYER.
LAYER="${LOGIT_LENS_LAYER:-10}"
export LOGIT_LENS_LAYER="${LAYER}"

LAYER_OUT="${PKG_DIR}/output/layer_${LAYER}"
mkdir -p "${LAYER_OUT}/figures_rank_diff"

CACHED_DISTRIBUTIONS="${CACHED_DISTRIBUTIONS:-${REPO_ROOT}/src/logit_lens_aim1_AND/output/layer_${LAYER}/extracted_distributions_top4096.jsonl}"
DISTRIBUTIONS_JSONL="${DISTRIBUTIONS_JSONL:-${LAYER_OUT}/extracted_distributions_top${LOGIT_LENS_TOP_K}.jsonl}"
METRICS_JSONL="${METRICS_JSONL:-${LAYER_OUT}/composition_metrics_or_max.jsonl}"
METRICS_CSV="${METRICS_CSV:-${LAYER_OUT}/composition_metrics_or_max.csv}"
RANK_DIFF_NPZ="${RANK_DIFF_NPZ:-${LAYER_OUT}/rank_diff_summary_q05_q95.npz}"
RANK_DIFF_FIG_DIR="${RANK_DIFF_FIG_DIR:-${LAYER_OUT}/figures_rank_diff}"

if [[ ! -f "${CACHED_DISTRIBUTIONS}" ]]; then
  echo "AND cache missing: ${CACHED_DISTRIBUTIONS}" >&2
  echo "Run logit_lens_aim1_AND for layer ${LAYER} first." >&2
  exit 1
fi

ARGS=(
  --cached-distributions "${CACHED_DISTRIBUTIONS}"
  --distributions-jsonl "${DISTRIBUTIONS_JSONL}"
  --metrics-jsonl "${METRICS_JSONL}"
  --metrics-csv "${METRICS_CSV}"
  --model-id "${LOGIT_LENS_MODEL}"
  --top-k "${LOGIT_LENS_TOP_K}"
  --layer-index "${LAYER}"
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

python -m logit_lens_aim1_OR.experiment "${ARGS[@]}"

echo "Done. Distributions: ${DISTRIBUTIONS_JSONL}"
echo "Metrics CSV: ${METRICS_CSV}"

PLOT_ARGS=(
  --layer-index "${LAYER}"
  --distributions-jsonl "${DISTRIBUTIONS_JSONL}"
  --summary-npz "${RANK_DIFF_NPZ}"
  --out-dir "${RANK_DIFF_FIG_DIR}"
  --recompute
)

if [[ -n "${MAX_PAIRS:-}" ]]; then
  PLOT_ARGS+=(--max-pairs "${MAX_PAIRS}")
fi

if [[ -n "${CATEGORIES:-}" ]]; then
  # shellcheck disable=SC2206
  CAT_ARR=(${CATEGORIES})
  PLOT_ARGS+=(--categories "${CAT_ARR[@]}")
fi

python -m logit_lens_aim1_OR.plot_rank_diffs "${PLOT_ARGS[@]}"

echo "Rank-diff summary: ${RANK_DIFF_NPZ}"
echo "Rank-diff figures: ${RANK_DIFF_FIG_DIR}/rank_diff_cat*.png"
