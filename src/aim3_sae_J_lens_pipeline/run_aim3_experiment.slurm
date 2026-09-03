#!/bin/bash
#SBATCH --job-name=ect_aim3_sae_j
#SBATCH --partition=gpu
#SBATCH --constraint=A100
#SBATCH --output=logs/ect_aim3_sae_j_%j.out
#SBATCH --error=logs/ect_aim3_sae_j_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

# Aim-3: Gemma-2-2B + Gemma Scope SAE (layer 14) + truncated J-lens Jacobian,
# then CPU tropical distances / Neighbor-Joining / triangle inequality.
#
# Requires the Aim-3 corpus:
#   src/aim3_data_prep/data/tropical_geometry_corpus.json
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch src/aim3_sae_J_lens_pipeline/run_aim3_experiment.slurm
#
# Smoke test:
#   sbatch --export=ALL,MAX_ITEMS=2,SAE_LAYER=14 \
#     src/aim3_sae_J_lens_pipeline/run_aim3_experiment.slurm
#
# Extract only (skip offline analysis):
#   sbatch --export=ALL,EXTRACT_ONLY=1 \
#     src/aim3_sae_J_lens_pipeline/run_aim3_experiment.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PKG_DIR="${REPO_ROOT}/src/aim3_sae_J_lens_pipeline"
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

SAE_ID="${SAE_ID:-layer_${SAE_LAYER}/width_${SAE_WIDTH}/canonical}"
LAYER_OUT="${PKG_DIR}/output/layer_${SAE_LAYER}"
mkdir -p "${LAYER_OUT}/tensors"

CORPUS="${CORPUS:-${REPO_ROOT}/src/aim3_data_prep/data/tropical_geometry_corpus.json}"
if [[ ! -f "${CORPUS}" ]]; then
  echo "Aim-3 corpus missing: ${CORPUS}" >&2
  echo "Run: sbatch src/aim3_data_prep/run_generate_aim3_corpus.slurm" >&2
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
  --top-k-sae "${AIM3_TOP_K_SAE:-256}"
  --top-k-vocab "${AIM3_TOP_K_VOCAB:-32}"
  --top-k-resid "${AIM3_TOP_K_RESID:-256}"
)

if [[ -n "${MAX_ITEMS:-}" ]]; then
  ARGS+=(--max-items "${MAX_ITEMS}")
fi
if [[ "${EXTRACT_ONLY:-0}" == "1" ]]; then
  ARGS+=(--extract-only)
fi
if [[ "${ANALYZE_ONLY:-0}" == "1" ]]; then
  ARGS+=(--analyze-only)
fi
if [[ "${NO_JACOBIAN:-0}" == "1" ]]; then
  ARGS+=(--no-jacobian)
fi

echo "=== Aim-3 SAE + J-lens tropical geometry ==="
echo "corpus=${CORPUS}"
echo "layer=${SAE_LAYER} sae_id=${SAE_ID}"
python -m aim3_sae_J_lens_pipeline.experiment "${ARGS[@]}"

echo "Done."
echo "Tensors:  ${LAYER_OUT}/tensors/"
echo "Manifest: ${LAYER_OUT}/extract_manifest.jsonl"
echo "Summary:  ${LAYER_OUT}/tropical_analysis_summary.json"
