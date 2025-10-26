python main.py \
    --mode prepare_sfttrain_data \
    --test_split_pct 0.01 \
    --context_length 256 \
    --data_store_dir "data/datasets/alpaca" \
    --hf_cache_dir "data/hf_cache" \
    --dataset_split_seed 42 \
    --num_workers 2 \
    --hf_model_name "answerdotai/ModernBERT-large"