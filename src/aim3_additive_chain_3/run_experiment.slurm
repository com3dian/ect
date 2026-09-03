#!/bin/bash
#SBATCH --job-name=ect_aim3_j3
#SBATCH --partition=gpu
#SBATCH --constraint=A100
#SBATCH --output=logs/ect_aim3_j3_%j.out
#SBATCH --error=logs/ect_aim3_j3_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

# Aim-3 v3: shared-trajectory additive chain in J-lens space (no SAE).
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch src/aim3_additive_chain_3/run_experiment.slurm
#
# Smoke:
#   sbatch --export=ALL,MAX_ITEMS=4,AIM3_TYPES=A \
#     src/aim3_additive_chain_3/run_experiment.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PKG_DIR="${REPO_ROOT}/src/aim3_additive_chain_3"
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
export PYTHONUNBUFFERED=1
export SAE_MODEL="${SAE_MODEL:-google/gemma-2-2b}"
export SAE_LAYER="${SAE_LAYER:-14}"
export SAE_DEVICE="${SAE_DEVICE:-cuda}"
export SAE_DTYPE="${SAE_DTYPE:-bfloat16}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

CORPUS="${CORPUS:-${REPO_ROOT}/src/aim3_data_prep/data/tropical_geometry_corpus.json}"
if [[ ! -f "${CORPUS}" ]]; then
  echo "Corpus missing: ${CORPUS}" >&2
  exit 1
fi

ARGS=(
  --corpus "${CORPUS}"
  --model-id "${SAE_MODEL}"
  --layer-index "${SAE_LAYER}"
  --device "${SAE_DEVICE}"
  --dtype "${SAE_DTYPE}"
  --output-dir "${PKG_DIR}/output"
  --types "${AIM3_TYPES:-A,B,C}"
  --top-k-vocab "${AIM3_TOP_K_VOCAB:-32}"
  --top-k-resid "${AIM3_TOP_K_RESID:-256}"
)

if [[ -n "${MAX_ITEMS:-}" ]]; then
  ARGS+=(--max-items "${MAX_ITEMS}")
fi

echo "=== Aim-3 shared-trajectory J-lens v3 ==="
python -m aim3_additive_chain_3.experiment "${ARGS[@]}"

echo "=== Plot Type A ==="
python -m aim3_additive_chain_3.plot_type_a \
  --layer-index "${SAE_LAYER}" \
  --output-dir "${PKG_DIR}/output"

echo "Done."
echo "Distances: ${PKG_DIR}/output/layer_${SAE_LAYER}/j_trajectory_distances.jsonl"
echo "Summary:   ${PKG_DIR}/output/layer_${SAE_LAYER}/j_trajectory_summary.json"
echo "Figure:    ${PKG_DIR}/output/layer_${SAE_LAYER}/figures/type_a_j_lens_trajectory.png"
