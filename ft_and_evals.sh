#!/bin/bash
set -e

# ====== Args ======
if [ -z "$1" ]; then
  echo "Usage: $0 <ft_dataset_size>"
  echo "Example: $0 ft_10k"
  exit 1
fi

FT_SIZE="$1"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL_DIR="${SCRIPT_DIR}/../evals/synllama/models/${FT_SIZE}_merged_synllama_r4_kqv"
mkdir -p "${MODEL_DIR}"

cd "${SCRIPT_DIR}"

# finetune
source .synft/bin/activate # uv environment
CUDA_VISIBLE_DEVICES="" python3 -m axolotl.cli.preprocess "${SCRIPT_DIR}/synllama/llm/sft/synllama_sft_runpod.yml"
CUDA_VISIBLE_DEVICES=0 accelerate launch -m axolotl.cli.train "${SCRIPT_DIR}/synllama/llm/sft/synllama_sft_runpod.yml"
CUDA_VISIBLE_DEVICES=0 python -m axolotl.cli.merge_lora "${SCRIPT_DIR}/synllama/llm/sft/synllama_sft_runpod.yml" --lora_model_dir="${MODEL_DIR}"

# evaluate
deactivate # uv
conda activate synllama_env
bash ./run_evals_runpod.sh "${FT_SIZE}"