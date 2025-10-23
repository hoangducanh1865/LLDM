import os
import argparse
class Config:
    TOKENIZER_MODEL_NAME='answerdotai/ModernBERT-base'
    @staticmethod
    def create_new_parser():
        parser=argparse.ArgumentParser()
        return parser
    @staticmethod
    def add_dataset_argument(parser):
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
        parser.add_argument('--batch_size',type=int,default=1000)