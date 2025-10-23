from tokenizers.processors import TemplateProcessing
from transformers import AutoTokenizer
from src.config import Config
class Tokenizer:
    @staticmethod
    def get_tokenizer(model_name=Config.TOKENIZER_MODEL_NAME,
                    bos_token='<BOS>',
                    eos_token='<EOS>',
                    start_token='<START_ID>',
                    end_token='<END_ID>',
                    eot_token='<EOT_ID>'):
        tokenizer=AutoTokenizer.from_pretrained(model_name)
        special_tokens={
            'bos_token':bos_token,
            'eos_token':eos_token,
            'additional_special_tokens':[start_token,end_token,eot_token]
        }
        tokenizer.add_special_tokens(special_tokens)
        
        # @QUESTION
        tokenizer.cls_token=bos_token 
        tokenizer.pad_token=eos_token
        
        tokenizer._tokenizer.post_processor=TemplateProcessing(
            single=f'{bos_token} $A {eos_token}',
            special_tokens=[
                (bos_token,tokenizer.bos_token_id),
                (eos_token,tokenizer.eos_token_id)
            ]
        )
        
        # Chat Template for SFT 
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "{{ bos_token if loop.first else '' }}"
            f"{{{{ '{start_token}' + message['role'] + '{end_token}' }}}}\n"
            "{{ message['content'] }}"
            f"{{{{ '{eot_token}' if message['role'] == 'user' else eos_token }}}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            f"{{{{ '{start_token}' + 'assistant' + '{end_token}' }}}}"
            "{% endif %}"
        )

        return tokenizer
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
'''if __name__=='__main__':
    test_tokenizer()'''