import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForMaskedLM, ModernBertForMaskedLM, get_scheduler
from accelerate import Accelerator
from tqdm import tqdm
from datasets import load_from_disk
from safetensors.torch import load_file
from src.config import Config
from src.data_manager import DataManager
from src.tokenizer import Tokenizer
from src.trainer import Trainer
def test_tokenizer():
    tokenizer=Tokenizer.get_tokenizer()
    text='Hello World'
    ids=tokenizer(text,padding=True,return_tensors='pt')['input_ids'][0]
    decoded=tokenizer.decode(ids,skip_special_tokens=False)
    messages=[
        {'role':'user','content':'What is the capital of Vietnam?'},
        {'role':'assistant','content':'Hanoi'}
    ]
    encoded=tokenizer.apply_chat_template(messages,tokenize=True,add_special_tokens=True)
    decoded=tokenizer.decode(encoded,skip_special_tokens=False)
    print(decoded)
    messages=[
        {'role':'user','content':'What is AI?'}
    ]
    encoded=tokenizer.apply_chat_template(messages,tokenize=True,add_special_tokens=True,add_generation_prompt=True)
    decoded=tokenizer.decode(encoded,skip_special_tokens=False)
    print(decoded)
def load_pretrain_dataset():
    parser=Config.create_new_parser()
    Config.add_pre_train_dataset_argument(parser)
    args=parser.parse_args()
    DataManager.prepare_data(args)
    start_time=time.time()
    data=load_from_disk(args.data_store_dir)
    end_time=time.time()
    print(f'Time to load dataset: {end_time-start_time}')
    print('Data:')
    print(data)
def pretrain():
    parser=Config.create_new_parser()
    Config.add_pre_training_argument(parser)
    args=parser.parse_args()
    experiment_dir=os.path.join(args.working_dir,args.experiment_name)
    accelerator=Accelerator(project_dir=experiment_dir,
                            log_with='wandb' if args.log_wandb else None)
    if args.log_wandb:
        accelerator.init_trackers(args.experiment_name)
    tokenizer=Tokenizer.get_tokenizer(args.hf_model_name)
    model=AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path=args.hf_model_name)
    model.resize_token_embeddings(len(tokenizer))
    model_parameters=[p for p in model.parameters() if p.requires_grad]
    params=sum([np.prod(p.size()) for p in model_parameters])
    accelerator.print('Number of parameters:',params)
    mini_batch_size=args.batch_size_per_gpu//args.gradient_accumulation_steps
    
    def collate_fn(batch):
        tokens=torch.stack([torch.tensor(b['input_ids'],dtype=torch.long) for b in batch]) # Every sample is in the same length
        return {'input_ids':tokens}
    
    tokenized_data=load_from_disk(args.dataset_dir)
    train_dataloader=DataLoader(tokenized_data['train'],
                                batch_size=mini_batch_size,
                                collate_fn=collate_fn, # @QUESTION 
                                shuffle=True)
    eval_dataloader=DataLoader(tokenized_data['test'],
                                batch_size=mini_batch_size,
                                collate_fn=collate_fn, # @QUESTION 
                                shuffle=False)
    optimizer=torch.optim.AdamW(model.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    scheduler=get_scheduler(name=args.lr_scheduler_type,
                            optimizer=optimizer,
                            num_warmup_steps=args.num_warmup_steps*accelerator.num_processes,
                            num_training_steps=args.num_training_steps*accelerator.num_processes)
    loss_fn=nn.CrossEntropyLoss(reduction='none') # @QUESTION
    model,optimizer,train_dataloader,eval_dataloader,scheduler=accelerator.prepare(
        model,optimizer,train_dataloader,eval_dataloader,scheduler
    )
    trainer=Trainer(args=args,
                    accelerator=accelerator,
                    tokenizer=tokenizer,
                    model=model,
                    train_dataloader=train_dataloader,
                    eval_dataloader=eval_dataloader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    experiment_dir=experiment_dir)
    trainer.pre_train()
def sfttrain():
    parser=Config.create_new_parser()
    Config.add_sft_training_argument(parser)
    args=parser.parse_args()
    experiment_dir=os.path.join(args.working_dir,args.experiment_name)
    accelerator=Accelerator(project_dir=experiment_dir,
                            log_with='wandb' if args.log_wandb else None)
    if args.log_wandb:
        accelerator.init_trackers(args.experiment_name)
    tokenizer=Tokenizer.get_tokenizer(args.hf_model_name)
    model=ModernBertForMaskedLM.from_pretrained(pretrained_model_name_or_path=args.hf_model_name)
    model.resize_token_embeddings(len(tokenizer))
    
    # Load pretrained model
    state_dict=load_file(args.pretrained_checkpoint_path)
    model.load_state_dict(state_dict,strict=False)
    model.tie_weights()
    
    model_parameters=[p for p in model.parameters() if p.requires_grad]
    params=sum([np.prod(p.size()) for p in model_parameters])
    accelerator.print('Number of parameters:',params)
    mini_batch_size=args.batch_size_per_gpu//args.gradient_accumulation_steps
    
    tokenized_data=load_from_disk(args.dataset_dir)
    train_dataloader=DataLoader(tokenized_data['train'],
                                batch_size=mini_batch_size,
                                collate_fn=DataManager.SFTCollator(args.hf_model_name), # @QUESTION 
                                shuffle=True)
    eval_dataloader=DataLoader(tokenized_data['test'],
                                batch_size=mini_batch_size,
                                collate_fn=DataManager.SFTCollator(args.hf_model_name), # @QUESTION 
                                shuffle=False)
    optimizer=torch.optim.AdamW(model.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    scheduler=get_scheduler(name=args.lr_scheduler_type,
                            optimizer=optimizer,
                            num_warmup_steps=args.num_warmup_steps,
                            num_training_steps=args.num_training_steps)
    loss_fn=nn.CrossEntropyLoss(reduction='none') # @QUESTION
    model,optimizer,train_dataloader,eval_dataloader,scheduler=accelerator.prepare(
        model,optimizer,train_dataloader,eval_dataloader,scheduler
    )
    trainer=Trainer(args=args,
                    accelerator=accelerator,
                    tokenizer=tokenizer,
                    model=model,
                    train_dataloader=train_dataloader,
                    eval_dataloader=eval_dataloader,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    loss_fn=loss_fn,
                    experiment_dir=experiment_dir)
    trainer.sft_train()
def main():
    load_pretrain_dataset()
if __name__=='__main__':
    main()