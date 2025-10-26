python main.py \
    --mode inference \
    --pretrained_model_path "data/work_dir/sfttraining_on_dataset_alpaca/final_model/model.safetensors" \
    --seq_len 512 \
    --num_steps 512 \
    --hf_model_name "answerdotai/ModernBERT-base" \
    --prompt "What is artificial intelligence?" \
    --remasking_strategy "low_confidence" \
    --show_mask 1