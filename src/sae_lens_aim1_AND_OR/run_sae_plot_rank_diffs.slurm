#!/bin/bash
#SBATCH --job-name=ect_sae_rankplot
#SBATCH --partition=main
#SBATCH --output=logs/ect_sae_rankplot_%j.out
#SBATCH --error=logs/ect_sae_rankplot_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# CPU-only replot of SAE AND/OR rank diffs (requires distributions JSONL).
#
#   sbatch --export=ALL,SAE_LAYER=14 src/sae_lens_aim1_AND_OR/run_sae_plot_rank_diffs.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PKG_DIR="${REPO_ROOT}/src/sae_lens_aim1_AND_OR"
CONDA_ENV="${CONDA_ENV:-ect}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"

mkdir -p "${REPO_ROOT}/logs"

if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
else
  echo "conda.sh not found at ${CONDA_ROOT}/etc/profile.d/conda.sh" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export SAE_LAYER="${SAE_LAYER:-14}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

LAYER_OUT="${PKG_DIR}/output/layer_${SAE_LAYER}"
DIST_JSONL="${DIST_JSONL:-${LAYER_OUT}/sae_feature_distributions.jsonl}"

if [[ ! -f "${DIST_JSONL}" ]]; then
  echo "Missing ${DIST_JSONL}" >&2
  echo "Run run_sae_extract_and_plot.slurm first." >&2
  exit 1
fi

ARGS=(
  --layer-index "${SAE_LAYER}"
  --distributions-jsonl "${DIST_JSONL}"
  --ops and or
  --recompute
)

if [[ -n "${MAX_PAIRS:-}" ]]; then
  ARGS+=(--max-pairs "${MAX_PAIRS}")
fi

if [[ -n "${CATEGORIES:-}" ]]; then
  # shellcheck disable=SC2206
  CAT_ARR=(${CATEGORIES})
  ARGS+=(--categories "${CAT_ARR[@]}")
fi

python -m sae_lens_aim1_AND_OR.plot_rank_diffs "${ARGS[@]}"

echo "AND figures: ${LAYER_OUT}/figures_rank_diff/and/"
echo "OR figures:  ${LAYER_OUT}/figures_rank_diff/or/"
