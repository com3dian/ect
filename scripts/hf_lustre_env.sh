# Hugging Face cache on WUR lustre scratch (source from shell or SLURM).
# Usage: source scripts/hf_lustre_env.sh

export HF_LUSTRE_ROOT="${HF_LUSTRE_ROOT:-/lustre/scratch/WUR/AIN/lu087}"
export HF_HOME="${HF_HOME:-${HF_LUSTRE_ROOT}/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"

mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${HF_DATASETS_CACHE}"
