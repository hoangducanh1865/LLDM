import torch
import logging
from rich.live import Live
from rich.console import Console
from rich.progress import Progress,BarColumn,TextColumn,TimeElapsedColumn,TimeRemainingColumn
from rich.text import Text
from safetensors.torch import load_file
from transformers import AutoModelForMaskedLM
from src.tokenizer import Tokenizer
from src.config import Config
class Inferencer:
    def __init__(self,args):
        self.args=args
        self.device=Config.DEVICE
        self.load_model_and_tokenizer()
    def load_model_and_tokenizer(self):
        self.tokenizer=Tokenizer.get_tokenizer(self.args.hf_model_name)
        self.model=AutoModelForMaskedLM.from_pretrained(self.args.hf_model_name,device_map=self.device)
        self.model.resize_token_embeddings(len(self.tokenizer))
        state_dict=load_file(self.args.pretrained_model_path)
        self.model.load_state_dict(state_dict,strict=False)
        self.model.tie_weights()
        self.model.eval()
    def prepare_unconditional_tokens_for_inference(self):
        input_tokens=torch.full((1,self.args.seq_len),self.tokenizer.mask_token_id,dtype=torch.long,device=self.device) # @QUESTION: why pass mask_token_id here?
        mask=torch.ones((1,self.args.seq_len),dtype=torch.bool,device=self.device)
        attention_mask=torch.ones((1,self.args.seq_len),dtype=torch.long,device=self.device)
        return input_tokens,mask,attention_mask
    def prepare_conditional_tokens_for_inference(self):
        chat_template=[
            {'role':'user','content':self.args.prompt}
        ]
        tokenized=self.tokenizer.apply_chat_template(chat_template,
                                                tokenize=True,
                                                add_special_tokens=True,
                                                add_generation_prompt=True)
        prompt_tokens=torch.tensor(tokenized).to(self.device)
        input_tokens,mask,attention_mask=self.prepare_unconditional_tokens_for_inference()
        input_tokens[0,:len(prompt_tokens)]=prompt_tokens
        mask[0,:len(prompt_tokens)]=False
        return input_tokens,mask,attention_mask
    @torch.no_grad()
    def inference(self):
        if self.args.prompt is None:
            ### Prepare Unconditional Inference Inputs ###
            input_tokens, mask, attention_mask = self.prepare_unconditional_tokens_for_inference()
        else:
            ### Prepare Conditional Inference Inputs ###
            input_tokens, mask, attention_mask = self.prepare_conditional_tokens_for_inference()
        ### Nice Printing Stuff ##
        console = Console(highlight=False)

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            
            ### What Controls our Progress Bar ###
            task = progress.add_task("Generating...", total=self.args.num_steps)

            ### Get Timesteps for Inference ###
            times = torch.linspace(1, 0, self.args.num_steps + 1, device=self.device)

            with Live("", refresh_per_second=5, console=console) as live:
                for t, s in zip(times[:-1], times[1:]):

                    ### Compute Logits ###
                    logits = self.model(input_tokens, attention_mask=attention_mask).logits

                    ### Sample Gen Token from Masked Tokens ###
                    probs = torch.softmax(logits[mask], dim=-1)
                    input_tokens[mask] = torch.multinomial(probs, num_samples=1).squeeze(-1) # @STAR: in original LLaDA paper, their code are different from this one
                    ### All Tokens are Randomly Remasked ###
                    if self.args.remasking_strategy == "random":

                        ### For Every Position, sample a value betweewn 0 and 1 ###
                        remask_probs = torch.rand_like(mask, dtype=torch.float, device=self.device)

                        ### If less than proportion token is selected to be remasked ###
                        remask_probs = (remask_probs < s/t)

                        ### Only replace if our mask token was previous True and is again True ###
                        ### once a token is false (no more masking) it is here to stay! ###
                        mask = mask & remask_probs

                        ### Set those tokens back to mask ###
                        input_tokens[mask] = self.tokenizer.mask_token_id

                    ### Low confidence Tokens are Randomly Remasked ###
                    elif self.args.remasking_strategy == "low_confidence":
                        
                        ### Compute Probs for all Tokens ###
                        probs_all = torch.nn.functional.softmax(logits, dim=-1)

                        ### Get the probability of the actually selected token ###
                        ### probs_all: 1 x self.args.seq_len x vocab_size
                        ### input_tokens: 1 x self.args.seq_len
                        chosen_token_probs = torch.gather(probs_all, dim=-1, 
                                                          index=input_tokens.unsqueeze(-1)).squeeze(-1)
                        
                        ### Make sure to set all tokens already selected to not be remasked to again ###
                        ### not be selected to be remasked. We can just set them to 1 because we want ###
                        ### low confidence (prob) tokens to be replaced! (set False to 1) ###
                        chosen_token_probs[~mask] = 1.0

                        ### Compute Proportion of Tokens to Remask out of the tokens that are currently masked ###
                        num_to_remask = int((s/t) * mask.sum().item())

                        if num_to_remask > 0:

                            ### Find the lowest prob tokens ###
                            lowest_confidence_idx = torch.topk(chosen_token_probs, num_to_remask, largest=False).indices

                            ### Create a New Mask (where everything is set to False) ###
                            new_mask = torch.zeros_like(mask)

                            ### Set the lowest confidence tokens to be remasked ###
                            new_mask[0, lowest_confidence_idx] = True
                            mask = new_mask

                            ### Update our Input Tokens with Mask Tokens ###
                            input_tokens[mask] = self.tokenizer.mask_token_id
                    
                    if self.args.show_mask:
                        ### Get all of the Tokens ###
                        decoded_tokens = self.tokenizer.convert_ids_to_tokens(input_tokens[0])

                        ### Keep [MASK] tokens, drop all other special tokens ###
                        cleaned_tokens = []
                        for tok in decoded_tokens:
                            if tok == self.tokenizer.mask_token:  # keep mask tokens
                                cleaned_tokens.append(tok)
                            elif tok in self.tokenizer.all_special_tokens:  # drop all other specials
                                continue
                            else:
                                cleaned_tokens.append(tok)

                        ### Put all the tokens back together into a string ###
                        decoded_after = self.tokenizer.convert_tokens_to_string(cleaned_tokens)
                    
                    else:
                        decoded_after = self.tokenizer.batch_decode(input_tokens, skip_special_tokens=True)[0]

                    if self.args.prompt is None:
                        format_text = Inferencer.format_display_for_unconditional(decoded_after)
                    else:
                        ### Remove Prompt Text from Assistant Text ###
                        assistant_text = decoded_after.replace(self.args.prompt, "").strip()
                        ### Remove Keywords user and assistant ###
                        assistant_text = Inferencer.clean_text(assistant_text)
                        format_text = Inferencer.format_display_for_qa(self.args.prompt, assistant_text)
                    live.update(format_text)
                    progress.update(task, advance=1)
    @staticmethod
    def format_display_for_qa(user_text, assistant_text):
        output = Text()
        output.append("USER: ", style="bold green")
        output.append(user_text + "\n\n")
        output.append("ASSISTANT: ", style="bold cyan")
        output.append(assistant_text, style="white")
        return output
    @staticmethod
    def format_display_for_unconditional(gen_text):
        output = Text()
        output.append("Unconditional Generation: \n\n", style="bold green")
        output.append(gen_text, style="white")
        return output
    @staticmethod
    def clean_text(raw_text: str) -> str:
        return (
            raw_text.replace("user", "")
            .replace("assistant", "")
            .strip()
        )