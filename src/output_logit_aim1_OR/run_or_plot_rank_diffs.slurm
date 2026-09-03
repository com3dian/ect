#!/bin/bash
#SBATCH --job-name=ect_or_plot
#SBATCH --partition=main
#SBATCH --output=logs/ect_or_plot_%j.out
#SBATCH --error=logs/ect_or_plot_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Plot min / mean / max rank-diff curves for output-logit OR (CPU-only).
#
# Uses conda env "ect" (override with CONDA_ENV=...).
#
# Submit from repo root:
#   mkdir -p src/output_logit_aim1_OR/logs
#   sbatch src/output_logit_aim1_OR/run_or_plot_rank_diffs.slurm
#
# Smoke test (first 50 pairs):
#   sbatch --export=ALL,MAX_PAIRS=50 src/output_logit_aim1_OR/run_or_plot_rank_diffs.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PKG_DIR="${REPO_ROOT}/src/output_logit_aim1_OR"
CONDA_ENV="${CONDA_ENV:-ect}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"

mkdir -p "${PKG_DIR}/logs" "${PKG_DIR}/output/figures_rank_diff"

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
export LOGIT_LENS_TOP_K="${LOGIT_LENS_TOP_K:-8192}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

DISTRIBUTIONS_JSONL="${DISTRIBUTIONS_JSONL:-${PKG_DIR}/output/extracted_distributions_top${LOGIT_LENS_TOP_K}.jsonl}"
RANK_DIFF_NPZ="${RANK_DIFF_NPZ:-${PKG_DIR}/output/rank_diff_summary_q05_q95.npz}"
RANK_DIFF_FIG_DIR="${RANK_DIFF_FIG_DIR:-${PKG_DIR}/output/figures_rank_diff}"

if [[ ! -f "${DISTRIBUTIONS_JSONL}" ]]; then
  echo "Distributions file missing: ${DISTRIBUTIONS_JSONL}" >&2
  echo "Run run_or_experiment.slurm first." >&2
  exit 1
fi

PLOT_ARGS=(
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

python -m output_logit_aim1_OR.plot_rank_diffs "${PLOT_ARGS[@]}"

echo "Rank-diff summary: ${RANK_DIFF_NPZ}"
echo "Rank-diff figures: ${RANK_DIFF_FIG_DIR}/rank_diff_cat*.png"
