import torch
import time
from src.tokenizer import Tokenizer
from src.config import Config
from datasets import load_dataset,load_from_disk,concatenate_datasets
class DataManager:
    @staticmethod
    def prepare_data(args):
        context_length=args.context_length
        data_store_dir=args.data_store_dir
        cache_dir=args.hf_cache_dir
        tokenizer=Tokenizer.get_tokenizer(args.hf_model_name)
        print('Loading data...')
        if args.large_dataset:
            fw=load_dataset('HuggingFaceFW/fineweb',
                            name='sample-10BT',
                            split='train',
                            cache_dir=cache_dir,
                            num_proc=args.num_workers)
            fw_edu=load_dataset('HuggingFaceFW/fineweb-edu',
                                name='sample-10BT',
                                split='train',
                                cache_dir=cache_dir,
                                num_proc=args.num_workers)
            wiki=load_dataset('wikimedia/wikipedia',
                              '20231101.en',
                              split='train',
                              cache_dir=cache_dir,
                              num_proc=args.num_workers)
            fw=fw.remove_columns([col for col in fw.column_names if col!='text'])
            fw_edu=fw_edu.remove_columns([col for col in fw_edu.column_names if col!='text'])
            wiki=wiki.remove_columns([col for col in wiki.column_names if col!='text'])
            dataset=concatenate_datasets([fw,fw_edu,wiki])
        else:
            # try:
            dataset=load_dataset('manu/project_gutenberg',
                                split='en',
                                cache_dir=cache_dir,
                                num_proc=args.num_workers)
            dataset=dataset.remove_columns([col for col in dataset.column_names if col!='text'])
            # except Exception as e:
            #     print(f"Failed to load project_gutenberg dataset: {e}")
            #     print("Fall
            # ing back to alternative dataset...")
            #     dataset=load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', cache_dir=cache_dir)
            #     dataset=dataset.remove_columns([col for col in dataset.column_names if col!='text'])
        dataset=dataset.train_test_split(test_size=args.test_split_pct,seed=args.dataset_split_seed)
        
        # Tokenize dataset
        def compute_tokens(examples):
            '''tokenized=tokenizer(examples['text'],
                                return_attention_mask=True,
                                add_special_tokens=True,
                                max_length=None,
                                truncation=False)'''
            tokenized=tokenizer(examples['text'],
                                return_attention_mask=True,
                                add_special_tokens=True,
                                max_length=context_length,
                                truncation=True,
                                padding='max_length')
            
            return {
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask']
            }
            
            # Chunk text
            '''input_ids_list=[]
            for ids in tokenized['input_ids']:
                for i in range(0,len(ids),context_length):
                    chunk=ids[i:i+context_length]
                    if len(chunk)<context_length:
                        chunk=chunk+[tokenizer.pad_token_id]*(context_length-len(chunk))
                    input_ids_list.append(chunk)
            return {'input_ids':input_ids_list}'''
        
        # Tokenize dataset
        '''tokenized_data=dataset.map(
            compute_tokens,
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_workers,
            remove_columns='text'
        )'''
        # Tokenize dataset with proper batch size
        print(f'Tokenizing data...')
        tokenized_data=dataset.map(
            compute_tokens,
            batched=True,
            batch_size=args.batch_size,  # Small batch size
            num_proc=args.num_workers,
            remove_columns=['text']
        )
        
        # Save dataset
        print(f'Saving tokenized dataset to: {data_store_dir}')
        tokenized_data.save_to_disk(data_store_dir)
    @staticmethod
    def prepare_sfttrain_dataset(args):
        context_length=args.context_length
        data_store_dir=args.data_store_dir
        cache_dir=args.hf_cache_dir
        tokenizer=Tokenizer.get_tokenizer(args.hf_model_name)
        dataset=load_dataset('tatsu-lab/alpaca',
                             split='train',
                             cache_dir=cache_dir,
                             num_proc=args.num_workers)
        def apply_chat_template(query,response):
            return tokenizer.apply_chat_template([
                {'role':'user','content':query},
                {'role':'assistant','content':response}
            ],
            tokenize=True,
            add_special_tokens=True)
        def preprocess(example):
            # Check HuggingFace for more detail on dataset 'tatsu-lab/alpaca'
            instruction=example['instruction']
            input=example['input']
            output=example['output']
            if len(input)>0:
                instruction=instruction.replace('.','')+': '+input
            tokenized=apply_chat_template(instruction,output)
            return {'input_ids':tokenized,'length':len(tokenized)}
        dataset=dataset.remove_columns('text')
        dataset=dataset.train_test_split(test_size=args.test_split_pct,seed=args.dataset_split_seed)
        tokenized_data=dataset.map(
            preprocess,
            num_proc=args.num_workers,
            remove_columns=['instruction','input','output'] # No longer needed since we just need tokenized data
        )
        def keep_within_context(example):
            return example['length']<=context_length
        tokenized_data=tokenized_data.filter(keep_within_context,num_proc=args.num_workers) # @TODO: re-write in condition statements
        tokenized_data=tokenized_data.remove_columns('length')
        def get_answer_mask(example):
            tokenized=example['input_ids']
            query_mask=[]
            occurance=0
            is_answer=False
            for t in tokenized:
                is_eos=(t==tokenizer.convert_tokens_to_ids('<END_ID>'))
                if is_answer==False:
                    query_mask.append(0)
                else:
                    query_mask.append(1)
                if is_eos:
                    occurance+=1
                    if occurance==2:
                        is_answer=True 
            example['query_mask']=query_mask
            return example
        tokenized_data=tokenized_data.map(
            get_answer_mask,
            num_proc=args.num_workers
        )
        tokenized_data.save_to_disk(data_store_dir)
    @staticmethod
    def SFTCollator(model_name):
        tokenizer=Tokenizer.get_tokenizer(model_name)
        eos_token=tokenizer.eos_token_id
        
        # @QUESTION
        def collate_fn(batch):
            inputs=[torch.tensor(b['input_ids'],dtype=torch.long) for b in batch]
            query_masks=[torch.tensor(b['query_mask']) for b in batch]
            inputs=torch.nn.utils.rnn.pad_sequence(inputs,padding_value=eos_token,batch_first=True) # Add padding token to the end of short sentences
            query_masks=torch.nn.utils.rnn.pad_sequence(query_masks,padding_value=1,batch_first=True) # Add mask token to the end of short sentences (token 1)
            return {'input_ids':inputs,'query_mask':query_masks}
        return collate_fn