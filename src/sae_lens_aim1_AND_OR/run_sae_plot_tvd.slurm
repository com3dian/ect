#!/bin/bash
#SBATCH --job-name=ect_sae_plot
#SBATCH --partition=main
#SBATCH --output=logs/ect_sae_plot_%j.out
#SBATCH --error=logs/ect_sae_plot_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Plot SAE AND/OR TVD vs min/mean/max (CPU-only).
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch --export=ALL,SAE_LAYER=14 src/sae_lens_aim1_AND_OR/run_sae_plot_tvd.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PKG_DIR="${REPO_ROOT}/src/sae_lens_aim1_AND_OR"
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
export SAE_LAYER="${SAE_LAYER:-14}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

LAYER_OUT="${PKG_DIR}/output/layer_${SAE_LAYER}"
METRICS_CSV="${METRICS_CSV:-${LAYER_OUT}/sae_composition_metrics.csv}"
FIG_DIR="${FIG_DIR:-${LAYER_OUT}/figures_tvd}"

if [[ ! -f "${METRICS_CSV}" ]]; then
  echo "Metrics CSV missing: ${METRICS_CSV}" >&2
  echo "Run run_sae_experiment.slurm first." >&2
  exit 1
fi

ARGS=(
  --layer-index "${SAE_LAYER}"
  --metrics-csv "${METRICS_CSV}"
  --out-dir "${FIG_DIR}"
)

if [[ -n "${CATEGORIES:-}" ]]; then
  # shellcheck disable=SC2206
  CAT_ARR=(${CATEGORIES})
  ARGS+=(--categories "${CAT_ARR[@]}")
fi

python -m sae_lens_aim1_AND_OR.plot_tvd_comparison "${ARGS[@]}"

echo "Figures: ${FIG_DIR}/tvd_and_or_cat*.png"
echo "Summary: ${FIG_DIR}/tvd_and_or_summary.png"
