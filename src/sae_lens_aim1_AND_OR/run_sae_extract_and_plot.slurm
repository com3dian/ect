#!/bin/bash
#SBATCH --job-name=ect_sae_feats
#SBATCH --partition=gpu
#SBATCH --constraint=A100
#SBATCH --output=logs/ect_sae_feats_%j.out
#SBATCH --error=logs/ect_sae_feats_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

# Extract sparse SAE Top-K features, then plot AND/OR rank-diffs
# (same 3-panel style as logit / logit-lens).
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch --export=ALL,SAE_LAYER=14 src/sae_lens_aim1_AND_OR/run_sae_extract_and_plot.slurm
#
# Smoke:
#   sbatch --export=ALL,MAX_PAIRS=2,SAE_LAYER=14 \
#     src/sae_lens_aim1_AND_OR/run_sae_extract_and_plot.slurm

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
export SAE_MODEL="${SAE_MODEL:-google/gemma-2-2b}"
export SAE_RELEASE="${SAE_RELEASE:-gemma-scope-2b-pt-res-canonical}"
export SAE_WIDTH="${SAE_WIDTH:-16k}"
export SAE_LAYER="${SAE_LAYER:-14}"
export SAE_DEVICE="${SAE_DEVICE:-cuda}"
export SAE_DTYPE="${SAE_DTYPE:-bfloat16}"
export SAE_TOP_K_FEATURES="${SAE_TOP_K_FEATURES:-256}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

SAE_ID="${SAE_ID:-layer_${SAE_LAYER}/width_${SAE_WIDTH}/canonical}"
LAYER_OUT="${PKG_DIR}/output/layer_${SAE_LAYER}"
mkdir -p "${LAYER_OUT}"

CORPUS="${CORPUS:-${REPO_ROOT}/src/vocabulary_gen_aim1_AND/output/orthogonality_spectrum_corpus.jsonl}"
DIST_JSONL="${DIST_JSONL:-${LAYER_OUT}/sae_feature_distributions.jsonl}"

EXTRACT_ARGS=(
  --corpus "${CORPUS}"
  --model-id "${SAE_MODEL}"
  --sae-release "${SAE_RELEASE}"
  --sae-id "${SAE_ID}"
  --layer-index "${SAE_LAYER}"
  --device "${SAE_DEVICE}"
  --dtype "${SAE_DTYPE}"
  --top-k "${SAE_TOP_K_FEATURES}"
  --distributions-jsonl "${DIST_JSONL}"
)

if [[ -n "${MAX_PAIRS:-}" ]]; then
  EXTRACT_ARGS+=(--max-pairs "${MAX_PAIRS}")
fi

if [[ -n "${CATEGORIES:-}" ]]; then
  # shellcheck disable=SC2206
  CAT_ARR=(${CATEGORIES})
  EXTRACT_ARGS+=(--categories "${CAT_ARR[@]}")
fi

echo "=== Extract sparse SAE features ==="
python -m sae_lens_aim1_AND_OR.extract_feature_distributions "${EXTRACT_ARGS[@]}"

PLOT_ARGS=(
  --layer-index "${SAE_LAYER}"
  --distributions-jsonl "${DIST_JSONL}"
  --ops and or
  --recompute
)

if [[ -n "${MAX_PAIRS:-}" ]]; then
  PLOT_ARGS+=(--max-pairs "${MAX_PAIRS}")
fi

if [[ -n "${CATEGORIES:-}" ]]; then
  # shellcheck disable=SC2206
  CAT_ARR=(${CATEGORIES})
  PLOT_ARGS+=(--categories "${CAT_ARR[@]}")
fi

echo "=== Plot AND/OR rank diffs ==="
python -m sae_lens_aim1_AND_OR.plot_rank_diffs "${PLOT_ARGS[@]}"

echo "Distributions: ${DIST_JSONL}"
echo "AND figures: ${LAYER_OUT}/figures_rank_diff/and/rank_diff_cat*.png"
echo "OR figures:  ${LAYER_OUT}/figures_rank_diff/or/rank_diff_cat*.png"
