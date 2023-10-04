GPU_ID=0

LLM=gpt2-medium # change LLM name here !!!
LLM_DIR=./llm/${LLM}/ # change LLM dir here !!!

DATA_DIR=data/ # change data dir here !!!

SEED_SMAPLE_LIST='[1,2,3,4,5]'

# for gpt2 family model, input length is 1024
array1=(mpqa) # maxshot = 32
array2=(sst2) # maxshot = 16
array3=(subj cr mr) # maxshot = 8
array4=(rte cb sst5) # maxshot = 4
array5=(agnews) # maxshot = 2
array6=(dbpedia) # maxshot = 1

# # for other LLMs, input length is 2048
# array1=(mpqa) # maxshot = 64
# array2=(sst2 cr) # maxshot = 32
# array3=(mr subj) # maxshot = 16
# array4=(rte sst5) # maxshot = 8
# array5=(agnews cb) # maxshot = 4
# array6=(dbpedia) # maxshot = 1
# ===================================================================

# iteratively run 1-shot to max-shot for each dataset on validation set

for DATASET in sst2 sst5 dbpedia mr cr mpqa subj agnews rte cb; do
        if [[ "${array1[@]}" =~ "${DATASET}" ]]; then
                N_SHOT_LIST="[1,2,4,8,16,32]"
        elif [[ "${array2[@]}" =~ "${DATASET}" ]]; then
                N_SHOT_LIST="[1,2,4,8,16]"
        elif [[ "${array3[@]}" =~ "${DATASET}" ]]; then
                N_SHOT_LIST="[1,2,4,8]"
        elif [[ "${array4[@]}" =~ "${DATASET}" ]]; then
                N_SHOT_LIST="[1,2,4]"
        elif [[ "${array5[@]}" =~ "${DATASET}" ]]; then
                N_SHOT_LIST="[1,2]"
        elif [[ "${array6[@]}" =~ "${DATASET}" ]]; then
                N_SHOT_LIST="[1]"
        fi

        python3 icl.py \
                --gpu_id ${GPU_ID} \
                --llm_dir ${LLM_DIR} \
                --dataset ${DATASET} \
                --data_dir ${DATA_DIR} \
                --n_shot_list ${N_SHOT_LIST} \
                --seed_sample_list ${SEED_SMAPLE_LIST} \
                --output_dir ~/oraclek_logs/
done
