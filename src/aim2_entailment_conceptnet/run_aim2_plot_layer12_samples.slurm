#!/bin/bash
#SBATCH --job-name=ect_aim2_cn_L12
#SBATCH --partition=main
#SBATCH --output=logs/ect_aim2_cn_L12_%j.out
#SBATCH --error=logs/ect_aim2_cn_L12_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Per-sample layer-12 plots for ConceptNet Aim-2 results.
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch src/aim2_entailment_conceptnet/run_aim2_plot_layer12_samples.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PKG_DIR="${REPO_ROOT}/src/aim2_entailment_conceptnet"
CONDA_ENV="${CONDA_ENV:-ect}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
OUTPUT_DIR="${OUTPUT_DIR:-${PKG_DIR}/output}"
PLOT_LAYER="${PLOT_LAYER:-12}"
PLOT_ALPHA="${PLOT_ALPHA:-1.0}"

mkdir -p "${REPO_ROOT}/logs" "${OUTPUT_DIR}"

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
export MPLBACKEND="${MPLBACKEND:-Agg}"

CSV="${RESULTS_CSV:-${OUTPUT_DIR}/patching_results.csv}"
if [[ ! -f "${CSV}" ]]; then
  echo "Results CSV missing: ${CSV}" >&2
  exit 1
fi

ARGS=(
  --csv "${CSV}"
  --output-dir "${OUTPUT_DIR}"
  --layer "${PLOT_LAYER}"
)

if [[ "${PLOT_ALPHA}" != "all" ]]; then
  ARGS+=(--alpha "${PLOT_ALPHA}")
fi

python -m aim2_entailment_conceptnet.plot_layer_samples "${ARGS[@]}"

echo "Figures: ${OUTPUT_DIR}/figures/layer${PLOT_LAYER}_per_sample/"
