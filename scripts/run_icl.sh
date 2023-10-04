GPU_ID=0

LLM=gpt2-medium # change LLM name here !!!
LLM_DIR=./llm/${LLM}/ # change LLM dir here !!!

DATA_DIR=data/ # change data dir here !!!

SEED_SMAPLE_LIST='[1,2,3,4,5]'

DATASET=subj
N_SHOT_LIST="[8]"
# ===================================================================


python3 icl.py \
        --gpu_id ${GPU_ID} \
        --llm_dir ${LLM_DIR} \
        --dataset ${DATASET} \
        --data_dir ${DATA_DIR} \
        --n_shot_list ${N_SHOT_LIST} \
        --seed_sample_list ${SEED_SMAPLE_LIST} \
        --output_dir ~/icl_logs/

