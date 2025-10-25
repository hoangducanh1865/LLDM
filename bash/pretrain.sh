accelerate launch pretrain.py \
--experiment_name "LDM_pretraining_on_large_dataset" \
--working_directory "data/work_dir" \
--hf_model_name "answerdotai/ModernBERT-base" \
--path_to_prepped_data "data/datasets/modernbert_large_dataset" \
--num_training_steps 100000 \
--per_gpu_batch_size 64 \
--gradient_accumulation_steps 4
