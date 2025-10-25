import os
import torch
import argparse
class Config:
    DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
    TOKENIZER_MODEL_NAME='answerdotai/ModernBERT-base'
    @staticmethod
    def create_new_parser():
        parser=argparse.ArgumentParser()
        return parser
    @staticmethod
    def add_pre_train_dataset_argument(parser):
        parser.add_argument('--test_split_pct',type=float,default=0.005,
                            help='Percentage of data that you want to use for train/test split')
        parser.add_argument('--context_length',type=int,default=1024)
        parser.add_argument('--data_store_dir',type=str,default=os.path.join('data','datasets','modernbert_large_dataset'),
                            help='Final tokenized dataset direction')
        parser.add_argument('--hf_cache_dir',type=str,default=os.path.join('data','hf_cache'))
        parser.add_argument('--dataset_split_seed',type=int,default=42)
        parser.add_argument('--num_workers',type=int,default=16)
        parser.add_argument('--hf_model_name',type=str,default='answerdotai/ModernBERT-base')
        parser.add_argument('--large_dataset',type=int,default=0,choices=[0,1])
        parser.add_argument('--batch_size',type=int,default=100)
    @staticmethod
    def add_pre_training_argument(parser):
        parser.add_argument('--experiment_name',type=str,default='LDM_pre_training_on_large_dataset')
        parser.add_argument('--working_dir',type=str,default=os.path.join('data','work_dir'))
        parser.add_argument('--hf_model_name',type=str,default='answerdotai/ModernBERT-base')
        parser.add_argument('--dataset_dir',type=str,default=os.path.join('data','datasets','modernbert_large_dataset'))
        parser.add_argument('--num_workers',type=int,default=8)
        parser.add_argument('--batch_size_per_gpu',type=int,default=16)
        parser.add_argument('--gradient_accumulation_steps',type=int,default=1,
                            help='Splits batch_size_per_gpu by gradient_accumulation_steps')
        parser.add_argument('--num_training_steps',type=int,default=100000)
        parser.add_argument('--max_grad_norm',type=float,default=1.0,
                            help='Max gradient norm used for stableizing traing with gradient clipping')
        parser.add_argument('--lr_scheduler_type',type=str,default='cosine')
        parser.add_argument('--num_warmup_steps',type=int,default=1000,
                            help='Number of steps for the warmup in the lr scheduler')
        parser.add_argument("--logging_steps",type=int,default=1,
                            help="Number of iterations for every log of metrics to wandb") # @QUESTION: what is the difference between this argument and argument 'gradient_accumulation_steps'?
        parser.add_argument("--evaluation_interval",type=int,default=2500,
                            help="Number of iterations for every evaluation and plotting")
        parser.add_argument("--checkpoint_interval",type=int,default=2500,
                            help="Number of iterations for checkpointing")
        parser.add_argument("--learning_rate",type=float,default=5e-5,
                            help="Max learning rate for all Learning Rate Schedulers")
        parser.add_argument("--weight_decay",type=float,default=0.01,
                            help="Weight decay constant for AdamW optimizer")
        parser.add_argument("--log_wandb",type=int,default=0,choices=[0,1],
                            help="Flag to enable logging to WanDB")
        return parser
    @staticmethod
    def add_sft_train_dataset_argument(parser):
        parser.add_argument('--test_split_pct',type=float,default=0.005,
                            help='Percentage of data that you want to use for train/test split')
        parser.add_argument('--context_length',type=int,default=1024)
        parser.add_argument('--data_store_dir',type=str,default=os.path.join('data','datasets','modernbert_large_dataset'),
                            help='Final tokenized dataset direction')
        parser.add_argument('--hf_cache_dir',type=str,default=os.path.join('data','hf_cache'))
        parser.add_argument('--dataset_split_seed',type=int,default=42)
        parser.add_argument('--num_workers',type=int,default=16)
        parser.add_argument('--hf_model_name',type=str,default='answerdotai/ModernBERT-base')
        parser.add_argument('--large_dataset',type=int,default=0,choices=[0,1])
    @staticmethod
    def add_sft_training_argument(parser):
        parser.add_argument('--experiment_name',type=str,default='LDM_sft_training_on_large_dataset')
        parser.add_argument('--working_dir',type=str,default=os.path.join('data','work_dir'))
        parser.add_argument('--pretrained_checkpoint_path',type=str,default=None)
        parser.add_argument('--hf_model_name',type=str,default='answerdotai/ModernBERT-base')
        parser.add_argument('--dataset_dir',type=str,default=os.path.join('data','datasets','sft_large_dataset'))
        parser.add_argument('--num_workers',type=int,default=8)
        parser.add_argument('--batch_size_per_gpu',type=int,default=16)
        parser.add_argument('--gradient_accumulation_steps',type=int,default=1,
                            help='Splits batch_size_per_gpu by gradient_accumulation_steps')
        parser.add_argument('--num_training_steps',type=int,default=100000)
        parser.add_argument('--max_grad_norm',type=float,default=1.0,
                            help='Max gradient norm used for stableizing traing with gradient clipping')
        parser.add_argument('--lr_scheduler_type',type=str,default='cosine')
        parser.add_argument('--num_warmup_steps',type=int,default=1000,
                            help='Number of steps for the warmup in the lr scheduler')
        parser.add_argument("--logging_steps",type=int,default=1,
                            help="Number of iterations for every log of metrics to wandb")
        parser.add_argument("--evaluation_interval",type=int,default=2500,
                            help="Number of iterations for every evaluation and plotting")
        parser.add_argument("--checkpoint_interval",type=int,default=2500,
                            help="Number of iterations for checkpointing")
        parser.add_argument("--learning_rate",type=float,default=5e-5,
                            help="Max learning rate for all Learning Rate Schedulers")
        parser.add_argument("--weight_decay",type=float,default=0.01,
                            help="Weight decay constant for AdamW optimizer")
        parser.add_argument("--log_wandb",type=int,default=0,choices=[0,1],
                            help="Flag to enable logging to WanDB")
        return parser