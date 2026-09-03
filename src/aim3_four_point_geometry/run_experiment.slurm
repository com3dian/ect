#!/bin/bash
#SBATCH --job-name=ect_aim3_fp
#SBATCH --partition=gpu
#SBATCH --constraint=A100
#SBATCH --output=logs/ect_aim3_fp_%j.out
#SBATCH --error=logs/ect_aim3_fp_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

# Aim-3 four-point geometry (Type A/B/C). Default: Gemma-2-2B on this cluster.
# Run 9B elsewhere: SAE_MODEL=google/gemma-2-9b sbatch ...
#
#   mkdir -p logs
#   sbatch src/aim3_four_point_geometry/run_experiment.slurm
#   sbatch --export=ALL,MAX_ITEMS=8,AIM3_TYPES=A \
#     src/aim3_four_point_geometry/run_experiment.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PKG_DIR="${REPO_ROOT}/src/aim3_four_point_geometry"
CONDA_ENV="${CONDA_ENV:-ect}"
CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"

export SAE_MODEL="${SAE_MODEL:-google/gemma-2-2b}"
MODEL_SLUG="${SAE_MODEL##*/}"
OUTPUT_DIR="${PKG_DIR}/output_${MODEL_SLUG}"

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
export PYTHONUNBUFFERED=1
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/hf_lustre_env.sh"
echo "HF_HOME=${HF_HOME}"
export SAE_DEVICE="${SAE_DEVICE:-cuda}"
export SAE_DTYPE="${SAE_DTYPE:-bfloat16}"
export SAE_LAYER="${SAE_LAYER:-14}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

CORPUS="${CORPUS:-${REPO_ROOT}/src/aim3_data_prep/data/tropical_geometry_corpus.json}"
if [[ ! -f "${CORPUS}" ]]; then
  echo "Corpus missing: ${CORPUS}" >&2
  exit 1
fi

echo "model=${SAE_MODEL}  output_dir=${OUTPUT_DIR}"

ARGS=(
  --corpus "${CORPUS}"
  --model-id "${SAE_MODEL}"
  --layer-index "${SAE_LAYER}"
  --device "${SAE_DEVICE}"
  --dtype "${SAE_DTYPE}"
  --output-dir "${OUTPUT_DIR}"
  --types "${AIM3_TYPES:-A,B,C}"
  --variants "${AIM3_VARIANTS:-natural,shuffled_middles,reversed}"
  --max-tokens "${AIM3_MAX_TARGET_TOKENS:-12}"
)

if [[ -n "${MAX_ITEMS:-}" ]]; then
  ARGS+=(--max-items "${MAX_ITEMS}")
fi

echo "=== Aim-3 four-point geometry ==="
python -m aim3_four_point_geometry.experiment "${ARGS[@]}"

echo "=== Plot Type A four-point ==="
python -m aim3_four_point_geometry.plot_type_a_four_point \
  --layer-index "${SAE_LAYER}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-id "${SAE_MODEL}"

echo "=== Plot Type B/C triangle ==="
python -m aim3_four_point_geometry.plot_type_bc_triangle \
  --layer-index "${SAE_LAYER}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-id "${SAE_MODEL}"

echo "Done."
echo "Distances: ${OUTPUT_DIR}/layer_${SAE_LAYER}/four_point_distances.jsonl"
echo "Summary:   ${OUTPUT_DIR}/layer_${SAE_LAYER}/four_point_summary.json"
echo "Figures:   ${OUTPUT_DIR}/layer_${SAE_LAYER}/figures/"
