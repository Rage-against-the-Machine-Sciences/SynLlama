# Test sets:
### 1K ChEMBL: ../evals/synllama/test_sets/1k_chembl.smi
### Enamine Testset from SynFormer ../evals/synllama/test_sets/1k_enamine_synformer.smi
### Zinc250k Testset (1k samples) from ReaSyn ../evals/synllama/test_sets/1k_zinc250k.smi
### Unseen Test set from 115 RXNs ../evals/synllama/test_sets/1k_test_unseen_bbs_115rxns.smi
### Unseen Test set from 91 RXNs ../evals/synllama/test_sets/1k_test_unseen_bbs_91rxns.smi
### Train-distribution subset from 115 RXNs ../evals/synllama/test_sets/1k_train_115rxns.smi
### Train-distribution subset from 91 RXNs ../evals/synllama/test_sets/1k_train_91rxns.smi

# Models:
# SynLlama-1B-2M-91rxns: The trained model for SynLlama-1B-2M using RXN Set 1.
# SynLlama-1B-2M-115rxns: The trained model for SynLlama-1B-2M using RXN Set 2.
# total (inference + reconstruct pathway + calculate metrics) combinations: 6 x 2 = 12

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/../synllama-data"
OUT_DIR="{SCRIPT_DIR}/../evals/synllama"

# ########################################## CLI ARGS

usage() {
    echo "Usage: $0 [--n_samples N] [--sample_mode MODE] [--seed S]"
    echo ""
    echo "Options:"
    echo "  --n_samples    Number of samples (default: 200)"
    echo "  --sample_mode  Sampling mode (default: high_only)"
    echo "  --seed         Random seed (default: 42)"
    exit 1
}

N_SAMPLES=200
SAMPLE_MODE=high_only
SEED=42

while [[ $# -gt 0 ]]; do
    case "$1" in
        --n_samples)   N_SAMPLES="$2";  shift 2 ;;
        --sample_mode) SAMPLE_MODE="$2"; shift 2 ;;
        --seed)        SEED="$2";       shift 2 ;;
        -h|--help)     usage ;;
        *) echo "Unknown argument: $1"; usage ;;
    esac
done

echo "Running evals with N_SAMPLES=${N_SAMPLES}, SAMPLE_MODE=${SAMPLE_MODE}, SEED=${SEED}"


MODELS=(
    "SynLlama-1B-2M-91rxns:91rxns"
    "SynLlama-1B-2M-115rxns:115rxns"
)

TEST_SETS=(
    "1k_chembl.smi:1k_chembl"
    "1k_enamine_synformer.smi:1k_enamine_synformer"
    "1k_zinc250k.smi:1k_zinc250k"
    # "1k_test_unseen_bbs_115rxns.smi:1k_test_unseen_bbs_115rxns"
    # "1k_test_unseen_bbs_91rxns.smi:1k_test_unseen_bbs_91rxns"
    # "1k_train_115rxns.smi:1k_train_115rxns"
    # "1k_train_91rxns.smi:1k_train_91rxns"
)

SMILES_DIR="${OUT_DIR}/test_sets"
MODEL_DIR="${DATA_DIR}/inference/model"
RESULTS_DIR="${OUT_DIR}/results/${SAMPLE_MODE}"
RECON_BASE="${DATA_DIR}/inference/reconstruction"

# cd once so that python -m module paths resolve correctly
cd "${SCRIPT_DIR}"

# prevent OpenBLAS from spawning excessive threads across multiprocessing workers
export OPENBLAS_NUM_THREADS=1

# ########################################## MAKE DIRS

for MODEL_ENTRY in "${MODELS[@]}"; do
    MODEL_NAME="${MODEL_ENTRY%%:*}"
    MODEL_TAG="${MODEL_ENTRY##*:}"
    for TS_ENTRY in "${TEST_SETS[@]}"; do
        TS_TAG="${TS_ENTRY##*:}"
        mkdir -p "${RESULTS_DIR}/synllama_1b_2m_${MODEL_TAG}_on_${TS_TAG}/logs"
    done
done

########################################## EVAL EXPERIMENTS

for MODEL_ENTRY in "${MODELS[@]}"; do
    MODEL_NAME="${MODEL_ENTRY%%:*}"
    MODEL_TAG="${MODEL_ENTRY##*:}"

    for TS_ENTRY in "${TEST_SETS[@]}"; do
        TS_FILE="${TS_ENTRY%%:*}"
        TS_TAG="${TS_ENTRY##*:}"

        RUN_NAME="synllama_1b_2m_${MODEL_TAG}_on_${TS_TAG}"
        RUN_DIR="${RESULTS_DIR}/${RUN_NAME}"
        LOGS="${RUN_DIR}/logs"

        echo "=========================================="
        echo "MODEL: ${MODEL_NAME}  |  TEST SET: ${TS_TAG}"
        echo "=========================================="

        python -m synllama.llm.parallel_inference \
            --model_path "${MODEL_DIR}/${MODEL_NAME}" \
            --smiles_path "${SMILES_DIR}/${TS_FILE}" \
            --save_path "${RUN_DIR}/${RUN_NAME}.pkl" \
            --sample_mode "${SAMPLE_MODE}" \
            --n_samples "${N_SAMPLES}" \
            --seed "${SEED}" \
            > >(tee "${LOGS}/parallel_inference.log") \
            2> >(tee "${LOGS}/parallel_inference_err.log" >&2)

        python -m steps.step_31_enamine_reconstruct \
            --llama_folder "${RUN_DIR}" \
            --embedding_path "${RECON_BASE}/${MODEL_TAG}/rxn_embeddings/" \
            --total_num_mols "${N_SAMPLES}" \
            --k 10 \
            --n_stacks 50 \
            --top_n_rows 50 \
            > >(tee "${LOGS}/step_31_enamine_reconstruct.log") \
            2> >(tee "${LOGS}/step_31_enamine_reconstruct_err.log" >&2)

        python -m steps.step_32_combined_stats_no_molport \
            --llama_folder "${RUN_DIR}" \
            --total_num_mols "${N_SAMPLES}" \
            > >(tee "${LOGS}/step_32_combined_stats.log") \
            2> >(tee "${LOGS}/step_32_combined_stats_err.log" >&2)

        # our custom metrics calculation
        python -m evals.diversity_eval \
            "${RUN_DIR}/enamine_reconstruct/${RUN_NAME}_enamine_reconstruct.csv" \
            --total "${N_SAMPLES}" \
            > >(tee "${LOGS}/diversity_eval.log") \
            2> >(tee "${LOGS}/diversity_eval_err.log" >&2)

    done
done
