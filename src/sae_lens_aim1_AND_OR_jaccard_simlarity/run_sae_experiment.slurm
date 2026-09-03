#!/bin/bash
#SBATCH --job-name=ect_sae_jacc
#SBATCH --partition=gpu
#SBATCH --constraint=A100
#SBATCH --output=logs/ect_sae_jacc_%j.out
#SBATCH --error=logs/ect_sae_jacc_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G

# SAE Jaccard set-theory AND/OR (thresholded active features vs ∩ / ∪).
#
# Submit from repo root:
#   mkdir -p logs
#   sbatch --export=ALL,SAE_LAYER=14 \
#     src/sae_lens_aim1_AND_OR_jaccard_simlarity/run_sae_experiment.slurm
#
# Smoke:
#   sbatch --export=ALL,MAX_PAIRS=2,SAE_LAYER=14 \
#     src/sae_lens_aim1_AND_OR_jaccard_simlarity/run_sae_experiment.slurm

set -euo pipefail

REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
PKG_DIR="${REPO_ROOT}/src/sae_lens_aim1_AND_OR_jaccard_simlarity"
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
export SAE_JACCARD_THRESHOLD="${SAE_JACCARD_THRESHOLD:-1.0}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

if [[ -z "${SAE_ID:-}" ]]; then
  if [[ "${SAE_RELEASE}" == *canonical* ]]; then
    SAE_ID="layer_${SAE_LAYER}/width_${SAE_WIDTH}/canonical"
  else
    SAE_ID="layer_${SAE_LAYER}/width_${SAE_WIDTH}/average_l0_${SAE_L0:-71}"
  fi
fi

LAYER_OUT="${PKG_DIR}/output/layer_${SAE_LAYER}"
mkdir -p "${LAYER_OUT}"

CORPUS="${CORPUS:-${REPO_ROOT}/src/vocabulary_gen_aim1_AND/output/orthogonality_spectrum_corpus.jsonl}"
METRICS_JSONL="${METRICS_JSONL:-${LAYER_OUT}/results_sae_jaccard.jsonl}"
METRICS_CSV="${METRICS_CSV:-${LAYER_OUT}/results_sae_jaccard.csv}"
METRICS_JSON="${METRICS_JSON:-${LAYER_OUT}/results_sae_jaccard.json}"
SUMMARY_CSV="${SUMMARY_CSV:-${LAYER_OUT}/category_summary.csv}"

ARGS=(
  --corpus "${CORPUS}"
  --model-id "${SAE_MODEL}"
  --sae-release "${SAE_RELEASE}"
  --sae-id "${SAE_ID}"
  --layer-index "${SAE_LAYER}"
  --device "${SAE_DEVICE}"
  --dtype "${SAE_DTYPE}"
  --threshold "${SAE_JACCARD_THRESHOLD}"
  --metrics-jsonl "${METRICS_JSONL}"
  --metrics-csv "${METRICS_CSV}"
  --metrics-json "${METRICS_JSON}"
  --category-summary-csv "${SUMMARY_CSV}"
)

if [[ -n "${MAX_PAIRS:-}" ]]; then
  ARGS+=(--max-pairs "${MAX_PAIRS}")
fi

if [[ -n "${CATEGORIES:-}" ]]; then
  # shellcheck disable=SC2206
  CAT_ARR=(${CATEGORIES})
  ARGS+=(--categories "${CAT_ARR[@]}")
fi

echo "=== SAE Jaccard experiment ==="
python -m sae_lens_aim1_AND_OR_jaccard_simlarity.experiment "${ARGS[@]}"

echo "=== Jaccard plots ==="
python -m sae_lens_aim1_AND_OR_jaccard_simlarity.plot_jaccard \
  --layer-index "${SAE_LAYER}" \
  --metrics-csv "${METRICS_CSV}" \
  --out-dir "${LAYER_OUT}/figures_jaccard"

echo "Done."
echo "Metrics: ${METRICS_CSV}"
echo "Figures: ${LAYER_OUT}/figures_jaccard/"
