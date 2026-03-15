# Test sets:
### 1K ChEMBL: ./synllama-data/inference/smiles/syn-planning/1k_chembl.smi
### Enamine Testset from SynFormer ./synllama-data/inference/smiles/syn-planning/1k_enamine_synformer.smi
### Unseen Test set from 115 RXNs ./synllama-data/inference/smiles/syn-planning/1k_test_unseen_bbs_115rxns.smi
### Unseen Test set from 91 RXNs ./synllama-data/inference/smiles/syn-planning/1k_test_unseen_bbs_91rxns.smi
### Train-distribution subset from 115 RXNs ./synllama-data/inference/smiles/syn-planning/1k_train_115rxns.smi
### Train-distribution subset from 91 RXNs ./synllama-data/inference/smiles/syn-planning/1k_train_91rxns.smi

# Models:
# SynLlama-1B-2M-91rxns: The trained model for SynLlama-1B-2M using RXN Set 1.
# SynLlama-1B-2M-115rxns: The trained model for SynLlama-1B-2M using RXN Set 2.
# total (inference + reconstruct pathway + calculate metrics) combinations: 6 x 2 = 12

MODELS=(
    # "SynLlama-1B-2M-91rxns:91rxns"
    "SynLlama-1B-2M-115rxns:115rxns"
)

TEST_SETS=(
    "1k_chembl.smi:1k_chembl"
    "1k_enamine_synformer.smi:1k_enamine_synformer"
    "test_zinc250k.smi:test_zinc250k"
    "1k_test_unseen_bbs_115rxns.smi:1k_test_unseen_bbs_115rxns"
    "1k_test_unseen_bbs_91rxns.smi:1k_test_unseen_bbs_91rxns"
    # "1k_train_115rxns.smi:1k_train_115rxns"
    # "1k_train_91rxns.smi:1k_train_91rxns"
)

SMILES_DIR="../synllama-data/inference/smiles/syn-planning"
MODEL_DIR="../synllama-data/inference/model"
RESULTS_DIR="../reproduce_benchmarks/synllama/inference_results/frozen_only/"
RECON_BASE="../synllama-data/inference/reconstruction"

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

    # MODEL: SynLlama-1B-2M-${MODEL_TAG}
    for TS_ENTRY in "${TEST_SETS[@]}"; do
        TS_FILE="${TS_ENTRY%%:*}"
        TS_TAG="${TS_ENTRY##*:}"

        RUN_NAME="synllama_1b_2m_${MODEL_TAG}_on_${TS_TAG}"
        RUN_DIR="${RESULTS_DIR}/${RUN_NAME}"
        LOGS="${RUN_DIR}/logs"

        ## TEST SET: ${TS_TAG}
        echo "=========================================="
        echo "MODEL: ${MODEL_NAME}  |  TEST SET: ${TS_TAG}"
        echo "=========================================="

        cd /workspace/SynLlama

        python -m synllama.llm.parallel_inference \
            --model_path "${MODEL_DIR}/${MODEL_NAME}" \
            --smiles_path "${SMILES_DIR}/${TS_FILE}" \
            --save_path "${RUN_DIR}/${RUN_NAME}.pkl" \
            --sample_mode frozen_only \
            > "${LOGS}/parallel_inference.log" 2> "${LOGS}/parallel_inference_err.log"

        python -m steps.step_31_enamine_reconstruct \
            --llama_folder "${RUN_DIR}" \
            --embedding_path "${RECON_BASE}/${MODEL_TAG}/rxn_embeddings/" \
            --total_num_mols 1000 \
            --k 10 \
            --n_stacks 50 \
            > "${LOGS}/step_31_enamine_reconstruct.log" 2> "${LOGS}/step_31_enamine_reconstruct_err.log"

        python -m steps.step_32_combined_stats_no_molport \
            --llama_folder "${RUN_DIR}" \
            --total_num_mols 1000 \
            > "${LOGS}/step_32_combined_stats.log" 2> "${LOGS}/step_32_combined_stats_err.log"

        # our custom metrics calculation
        python -m evals.diversity_eval \
            "${RUN_DIR}/enamine_reconstruct/${RUN_NAME}_enamine_reconstruct.csv" \
            --total 1000 \
            > "${LOGS}/diversity_eval.log" 2> "${LOGS}/diversity_eval_err.log"

    done
done
