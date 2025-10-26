python main.py --mode prepare_pretrain_data \
    --test_split_pct 0.005 \
    --context_length 256 \
    --data_store_dir "data/datasets/gutenberg" \
    --hf_cache_dir "data/hf_cache" \
    --dataset_split_seed 42 \
    --num_workers 2 \
    --hf_model_name "answerdotai/ModernBERT-large" \
    --large_dataset 0 \
    --batch_size 50