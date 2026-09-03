#!/bin/bash
#SBATCH --job-name=ect_aim3_traj
#SBATCH --partition=gpu
#SBATCH --constraint=A100
#SBATCH --output=logs/ect_aim3_traj_%j.out
#SBATCH --error=logs/ect_aim3_traj_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

# Aim-3 v2: shared-trajectory additive chain (Type A/B/C).
# One forward per item; SAE + logit-lens distances between waypoints.
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch src/aim3_addtive_chain_2/run_experiment.slurm
#
# Smoke:
#   sbatch --export=ALL,MAX_ITEMS=4,AIM3_TYPES=A \
#     src/aim3_addtive_chain_2/run_experiment.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PKG_DIR="${REPO_ROOT}/src/aim3_addtive_chain_2"
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
export SAE_RELEASE="${SAE_RELEASE:-gemma-scope-2b-pt-res-canonical}"
export SAE_WIDTH="${SAE_WIDTH:-16k}"
export SAE_LAYER="${SAE_LAYER:-14}"
export SAE_DEVICE="${SAE_DEVICE:-cuda}"
export SAE_DTYPE="${SAE_DTYPE:-bfloat16}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

SAE_ID="${SAE_ID:-layer_${SAE_LAYER}/width_${SAE_WIDTH}/canonical}"
CORPUS="${CORPUS:-${REPO_ROOT}/src/aim3_data_prep/data/tropical_geometry_corpus.json}"

if [[ ! -f "${CORPUS}" ]]; then
  echo "Corpus missing: ${CORPUS}" >&2
  exit 1
fi

ARGS=(
  --corpus "${CORPUS}"
  --model-id "${SAE_MODEL}"
  --sae-release "${SAE_RELEASE}"
  --sae-id "${SAE_ID}"
  --layer-index "${SAE_LAYER}"
  --device "${SAE_DEVICE}"
  --dtype "${SAE_DTYPE}"
  --output-dir "${PKG_DIR}/output"
  --types "${AIM3_TYPES:-A,B,C}"
)

if [[ -n "${MAX_ITEMS:-}" ]]; then
  ARGS+=(--max-items "${MAX_ITEMS}")
fi

echo "=== Aim-3 shared-trajectory additive chain v2 ==="
python -m aim3_addtive_chain_2.experiment "${ARGS[@]}"

echo "=== Plot Type A ==="
python -m aim3_addtive_chain_2.plot_type_a \
  --layer-index "${SAE_LAYER}" \
  --output-dir "${PKG_DIR}/output"

echo "Done."
echo "Distances: ${PKG_DIR}/output/layer_${SAE_LAYER}/trajectory_distances.jsonl"
echo "Summary:   ${PKG_DIR}/output/layer_${SAE_LAYER}/trajectory_summary.json"
echo "Figure:    ${PKG_DIR}/output/layer_${SAE_LAYER}/figures/type_a_shared_trajectory.png"
