#!/bin/bash
#SBATCH --job-name=ect_aim3_trop
#SBATCH --partition=main
#SBATCH --output=logs/ect_aim3_trop_%j.out
#SBATCH --error=logs/ect_aim3_trop_%j.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Offline tropical topology (CPU): Neighbor-Joining + triangle inequality.
# Requires tensors already extracted by run_aim3_experiment.slurm.
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch src/aim3_sae_J_lens_pipeline/run_aim3_analyze.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PKG_DIR="${REPO_ROOT}/src/aim3_sae_J_lens_pipeline"
CONDA_ENV="${CONDA_ENV:-ect}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
SAE_LAYER="${SAE_LAYER:-14}"

mkdir -p "${REPO_ROOT}/logs"

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
export PYTHONUNBUFFERED=1

LAYER_OUT="${PKG_DIR}/output/layer_${SAE_LAYER}"
if [[ ! -f "${LAYER_OUT}/extract_manifest.jsonl" ]]; then
  echo "No extract manifest at ${LAYER_OUT}/extract_manifest.jsonl" >&2
  echo "Run the GPU job first: sbatch src/aim3_sae_J_lens_pipeline/run_aim3_experiment.slurm" >&2
  exit 1
fi

python -m aim3_sae_J_lens_pipeline.analyze_tropical_topology \
  --layer-index "${SAE_LAYER}" \
  --output-dir "${PKG_DIR}/output"

echo "Done."
echo "Summary: ${LAYER_OUT}/tropical_analysis_summary.json"
echo "NJ:      ${LAYER_OUT}/neighbor_joining_newick.jsonl"
echo "Tri:     ${LAYER_OUT}/triangle_inequality.csv"
