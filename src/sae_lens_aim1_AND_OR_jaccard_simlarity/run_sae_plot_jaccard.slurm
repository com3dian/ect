#!/bin/bash
#SBATCH --job-name=ect_sae_jacc_plot
#SBATCH --partition=main
#SBATCH --output=logs/ect_sae_jacc_plot_%j.out
#SBATCH --error=logs/ect_sae_jacc_plot_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# CPU replot of SAE Jaccard box/bar charts.
#
#   sbatch --export=ALL,SAE_LAYER=14 \
#     src/sae_lens_aim1_AND_OR_jaccard_simlarity/run_sae_plot_jaccard.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PKG_DIR="${REPO_ROOT}/src/sae_lens_aim1_AND_OR_jaccard_simlarity"
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
METRICS_CSV="${METRICS_CSV:-${LAYER_OUT}/results_sae_jaccard.csv}"
FIG_DIR="${FIG_DIR:-${LAYER_OUT}/figures_jaccard}"

if [[ ! -f "${METRICS_CSV}" ]]; then
  echo "Metrics CSV missing: ${METRICS_CSV}" >&2
  exit 1
fi

ARGS=(--layer-index "${SAE_LAYER}" --metrics-csv "${METRICS_CSV}" --out-dir "${FIG_DIR}")
if [[ -n "${CATEGORIES:-}" ]]; then
  # shellcheck disable=SC2206
  CAT_ARR=(${CATEGORIES})
  ARGS+=(--categories "${CAT_ARR[@]}")
fi

python -m sae_lens_aim1_AND_OR_jaccard_simlarity.plot_jaccard "${ARGS[@]}"
echo "Figures: ${FIG_DIR}"
