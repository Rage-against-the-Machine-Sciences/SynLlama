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
DATA_DIR="${SCRIPT_DIR}/../synllama-data"
N_SAMPLES=200
SEED=42

EMBEDDINGS="${DATA_DIR}/inference/reconstruction/115rxns/rxn_embeddings"
SMILES="${DATA_DIR}/inference/smiles/syn-planning/1k_enamine_synformer.smi"

run_pipeline() {
  local MODEL="$1"
  local RUN_NAME="$2"
  local FT_SIZE="$3"
  local SKIP_INFERENCE="${4:-false}"

  local RUN_DIR="${SCRIPT_DIR}/../evals/synllama/results/${FT_SIZE}/sequential_medium_only/${RUN_NAME}"
  local LOGS="${RUN_DIR}/logs"
  mkdir -p "${LOGS}"

  echo "=== Running ${RUN_NAME} (${FT_SIZE}) ==="

  echo "--- Step 1: LLM inference (medium_only, sequential) ---"
  if [ "${SKIP_INFERENCE}" = "true" ]; then
    echo "  (skipped — using existing pkl)"
  else
    python -m synllama.llm.parallel_inference \
      --model_path "$MODEL" \
      --smiles_path "$SMILES" \
      --save_path "${RUN_DIR}/${RUN_NAME}.pkl" \
      --sample_mode medium_only \
      --n_samples "${N_SAMPLES}" \
      --seed "${SEED}" \
      --sequential \
      > >(tee "${LOGS}/parallel_inference.log") \
      2> >(tee "${LOGS}/parallel_inference_err.log" >&2)
  fi

  echo "--- Step 2: Filter raw output ---"
  python -m steps.step_30_0_benchmark_filter_raw_output \
    --llama_folder "${RUN_DIR}" \
    --rxn_mapping_path "${EMBEDDINGS}/reaction_smarts_map.pkl" \
    --fp_searcher_path "${DATA_DIR}/inference/reconstruction/115rxns/processed/fpindex.pkl" \
    --raw_output_only \
    > >(tee "${LOGS}/step_30_0.log") \
    2> >(tee "${LOGS}/step_30_0_err.log" >&2)

  echo "--- Step 3: Direct Enamine hits ---"
  python -m steps.step_30_1_molport_raw_reconstruct \
    --llama_folder "${RUN_DIR}" \
    --enamine_only \
    > >(tee "${LOGS}/step_30_1.log") \
    2> >(tee "${LOGS}/step_30_1_err.log" >&2)

  echo "--- Step 4: Enamine reconstruction ---"
  python -m steps.step_31_enamine_reconstruct \
    --llama_folder "${RUN_DIR}" \
    --embedding_path "${EMBEDDINGS}" \
    --total_num_mols "${N_SAMPLES}" \
    --k 5 \
    --n_stacks 25 \
    --top_n_rows 50 \
    > >(tee "${LOGS}/step_31_enamine_reconstruct.log") \
    2> >(tee "${LOGS}/step_31_enamine_reconstruct_err.log" >&2)

  echo "--- Step 5: Combined stats ---"
  python -m steps.step_32_combined_stats \
    --llama_folder "${RUN_DIR}" \
    --total_num_mols "${N_SAMPLES}" \
    > >(tee "${LOGS}/step_32_combined_stats.log") \
    2> >(tee "${LOGS}/step_32_combined_stats_err.log" >&2)

  echo "--- Step 6: Diversity eval ---"
  python -m evals.diversity_eval \
    "${RUN_DIR}/enamine_reconstruct/${RUN_NAME}_enamine_reconstruct.csv" \
    --total "${N_SAMPLES}" \
    > >(tee "${LOGS}/diversity_eval.log") \
    2> >(tee "${LOGS}/diversity_eval_err.log" >&2)

  echo "=== Done. Results in ${RUN_DIR} ==="
}

cd "${SCRIPT_DIR}"
export OPENBLAS_NUM_THREADS=1

BASE_MODEL="${DATA_DIR}/inference/model/SynLlama-1B-2M-115rxns"
FT_MODEL="${SCRIPT_DIR}/../evals/synllama/models/${FT_SIZE}_merged_synllama_r4_kqv"

# run_pipeline "${BASE_MODEL}" "synllama_base_${FT_SIZE}_on_1k_enamine_synformer" "${FT_SIZE}"
run_pipeline "${FT_MODEL}"   "synllama_merged_${FT_SIZE}_lr5e5_on_1k_enamine_synformer" "${FT_SIZE}"