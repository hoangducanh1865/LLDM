python main.py \
    --test_split_pct 0.005 \
    --context_length 1024 \
    --data_store_dir "data/datasets/modernbert_large_dataset" \
    --hf_cache_dir "data/hf_cache" \
    --dataset_split_seed 42 \
    --num_workers 1 \
    --hf_model_name "answerdotai/ModernBERT-large" \
    --large_dataset 0
