#!/bin/bash
#SBATCH --job-name=ect_aim2_plot
#SBATCH --partition=main
#SBATCH --output=logs/ect_aim2_plot_%j.out
#SBATCH --error=logs/ect_aim2_plot_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

# Replot Aim-2 patching CSV (CPU-only). Submit from repo root:
#   mkdir -p logs
#   sbatch src/aim2_entailment/run_aim2_plot.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PKG_DIR="${REPO_ROOT}/src/aim2_entailment"
CONDA_ENV="${CONDA_ENV:-ect}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
OUTPUT_DIR="${OUTPUT_DIR:-${PKG_DIR}/output}"

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

python -m aim2_entailment.plot_patching --csv "${CSV}" --output-dir "${OUTPUT_DIR}"

echo "Figures: ${OUTPUT_DIR}/figures/"
