for DATASET in sst2 sst5 dbpedia mr cr mpqa subj agnews rte cb; do
    python ./utils/preprocess.py \
        --data_dir ./data \
        --dataset ${DATASET}
done