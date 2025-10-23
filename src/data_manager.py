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
            fw_edu=fw_edu.remove_columns([col for col in fw.column_names if col!='text'])
            wiki=wiki.remove_columns([col for col in fw.column_names if col!='text'])
            dataset=concatenate_datasets([fw,fw_edu,wiki])
        else:
            dataset=load_dataset('manu/project_gutenberg',
                                 split='en',
                                 cache_dir=cache_dir,
                                 num_proc=args.num_workers)
            dataset=dataset.remove_columns([col for col in dataset.column_names if col!='text'])
        dataset=dataset.train_test_split(test_size=args.text_split_pct,seed=args.data_split_seed)
        
        # Tokenize dataset
        def compute_tokens(examples):
            tokenized=tokenizer(examples['text'],
                                return_attention_mask=True,
                                add_special_tokens=True,
                                max_length=None,
                                truncation=False)
            
            # Chunk text
            input_ids_list=[]
            for ids in tokenized['input_ids']:
                for i in range(0,len(ids),context_length):
                    chunk=ids[i:i+context_length]
                    if len(chunk)<context_length:
                        chunk=chunk+[tokenizer.pad_token_id]*(context_length-len(chunk))
                    input_ids_list.append(chunk)
            return {'input_ids':input_ids_list}
        
        # Tokenize dataset
        tokenized_data=dataset.map(
            compute_tokens,
            batched=True,
            batch_size=args.batch_size,
            num_proc=args.num_workers,
            remove_columns='text'
        )
        
        # Save dataset
        print(f'Saving tokenized dataset to: {data_store_dir}')
        tokenized_data.save_to_disk(data_store_dir)
if __name__=='__main__':
    parser=Config.create_new_parser()
    Config.add_dataset_argument(parser)
    args=parser.parse_args()
    DataManager.prepare_data(args)
    start_time=time.time()
    data=load_from_disk(args.data_store_dir)
    end_time=time.time()
    print(f'Time to load dataset: {end_time-start_time}')
    print('Data:')
    print(data)