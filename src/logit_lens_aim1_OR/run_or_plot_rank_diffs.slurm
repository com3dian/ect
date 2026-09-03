#!/bin/bash
#SBATCH --job-name=ect_or_lens_plot
#SBATCH --partition=main
#SBATCH --output=logs/ect_or_lens_plot_%j.out
#SBATCH --error=logs/ect_or_lens_plot_%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Plot min / mean / max rank-diff curves for middle-layer OR (CPU-only).
# Default: all completed layers 10 12 14 16. Override with LAYERS="10 12".
#
# Uses conda env "ect" (override with CONDA_ENV=...).
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch src/logit_lens_aim1_OR/run_or_plot_rank_diffs.slurm
#
# Single layer:
#   sbatch --export=ALL,LAYERS=10 src/logit_lens_aim1_OR/run_or_plot_rank_diffs.slurm
#
# Smoke test:
#   sbatch --export=ALL,LAYERS=10,MAX_PAIRS=50 src/logit_lens_aim1_OR/run_or_plot_rank_diffs.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PKG_DIR="${REPO_ROOT}/src/logit_lens_aim1_OR"
CONDA_ENV="${CONDA_ENV:-ect}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"

mkdir -p "${REPO_ROOT}/logs" "${PKG_DIR}/output"

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
export LOGIT_LENS_TOP_K="${LOGIT_LENS_TOP_K:-4096}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

# shellcheck disable=SC2206
LAYERS=(${LAYERS:-10 12 14 16})

for LAYER in "${LAYERS[@]}"; do
  LAYER_OUT="${PKG_DIR}/output/layer_${LAYER}"
  DISTRIBUTIONS_JSONL="${LAYER_OUT}/extracted_distributions_top${LOGIT_LENS_TOP_K}.jsonl"
  RANK_DIFF_NPZ="${LAYER_OUT}/rank_diff_summary_q05_q95.npz"
  RANK_DIFF_FIG_DIR="${LAYER_OUT}/figures_rank_diff"

  if [[ ! -f "${DISTRIBUTIONS_JSONL}" ]]; then
    echo "Skipping layer ${LAYER}: missing ${DISTRIBUTIONS_JSONL}" >&2
    continue
  fi

  mkdir -p "${RANK_DIFF_FIG_DIR}"
  echo "=== Plotting layer ${LAYER} ==="

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

  echo "Layer ${LAYER} summary: ${RANK_DIFF_NPZ}"
  echo "Layer ${LAYER} figures: ${RANK_DIFF_FIG_DIR}/rank_diff_cat*.png"
done

echo "Done plotting layers: ${LAYERS[*]}"
