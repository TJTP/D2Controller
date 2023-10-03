LLM=opt-30b # only 30b needs multi cards
LLM_DIR=./llm/${LLM}/ # change LLM dir here !!!
RTRV_MODEL=gpt2-xl # use gpt2-xl as retrieval model
RTRV_DIR=./llm/${RTRV_MODEL}/ # change RTRV dir here !!!

DIS_TYPE=iicscore

DATA_DIR=data/

SEED_SAMPLE_LIST="[1,2,3,4,5]"

GPU_ID=0

# ===================================================================
array1=(mpqa) # maxshot = 64
array2=(sst2 cr) # maxshot = 32
array3=(mr subj) # maxshot = 16
array4=(rte sst5) # maxshot = 8
array5=(agnews cb) # maxshot = 4
array6=(dbpedia) # maxshot = 1, no need to select

for DATASET in sst2 sst5 mr cr mpqa subj agnews rte cb; do

        if [[ "${array1[@]}" =~ "${DATASET}" ]]; then
                N_SHOT_LIST="[1,2,4,8,16,32,64]"
        elif [[ "${array2[@]}" =~ "${DATASET}" ]]; then
                N_SHOT_LIST="[1,2,4,8,16,32]"
        elif [[ "${array3[@]}" =~ "${DATASET}" ]]; then
                N_SHOT_LIST="[1,2,4,8,16]"
        elif [[ "${array4[@]}" =~ "${DATASET}" ]]; then
                N_SHOT_LIST="[1,2,4,8]"
        elif [[ "${array5[@]}" =~ "${DATASET}" ]]; then
                N_SHOT_LIST="[1,2,4]"
        fi
        
        CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed select_k.py \
                --multi_gpu 1 \
                --llm_dir ${LLM_DIR} \
                --gpu_id ${GPU_ID} \
                --rtrv_model_dir ${RTRV_DIR} \
                --dis_type ${DIS_TYPE} \
                --dataset ${DATASET} \
                --data_dir ${DATA_DIR} \
                --n_shot_list ${N_SHOT_LIST} \
                --seed_sample_list ${SEED_SAMPLE_LIST} \
                --output_dir ~/selectk_logs/
done
