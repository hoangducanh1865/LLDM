import os
import torch
from tqdm import tqdm
from src.config import Config
class Trainer:
    def __init__(self,args,accelerator,tokenizer,model,train_dataloader,eval_dataloader,optimizer,scheduler,loss_fn,experiment_dir):
        self.args=args
        self.accelerator=accelerator
        self.tokenizer=tokenizer
        self.model=model
        self.train_dataloader=train_dataloader
        self.eval_dataloader=eval_dataloader
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.loss_fn=loss_fn
        self.experiment_dir=experiment_dir
        self.device=Config.DEVICE
    def pretrain(self):
        train=True
        completed_steps=0
        progress_bar=tqdm(range(completed_steps,self.args.num_training_steps),disable=not self.accelerator.is_local_main_process) # Since there are multiple processes so we need to just display the main one
        while train:
            accumulate_step=0
            accumulate_loss=0
            for batch in self.train_dataloader:
                input_ids=batch['input_ids'].to(self.accelerator.device)
                
                # Attend to all tokens
                batch_size,seq_len=input_ids.shape
                attention_mask=torch.ones((batch_size,seq_len),dtype=torch.long,device=self.accelerator.device) # Since every tokens are attend so in here we use ones, not zeros
                
                # Random sample t to mask tokens of each line in batch
                t=torch.rand(batch_size,1,device=self.accelerator.device).expand(batch_size,seq_len).clamp_min(1e-5) # Clamp min so that every single number will not be zero
                # @STAR: for-loop form
                '''t=torch.zeros(batch_size,seq_len,device=self.accelerator.device)
                for i in range(batch_size):
                    for j in range(seq_len):
                        random_prob=torch.rand(1,device=self.accelerator.device).item() # Use item() to extract scalar value from a single-element tensor, without this, random_prob will be a tensor object, so we can not assign t[i, j] = random_prob since t[i, j] is scalar value
                        if random_prob<1e-5:
                            random_prob=1e-5
                        t[i,j]=random_prob'''
                        
                mask=torch.bernoulli(t).bool() # Decide which token to be masked
                
                # @QUESTION
                masked_input_ids=input_ids.masked_fill(mask,self.tokenizer.mask_token_id) # @QUESTION: where does mask_token_id come from?
                labels=input_ids.masked_fill(~mask,-100) 
                logits=self.model(input_ids=masked_input_ids,attention_mask=attention_mask)['logits']
                num_classes=logits.shape[-1]
                loss=self.loss_fn(logits.reshape(batch_size*seq_len,num_classes),
                            labels.flatten())
                loss=loss.reshape(batch_size,seq_len)/t # Just for fairness, if t increase then loss decrease, if i decrease then loss increase
                loss=loss.mean()
                loss=loss/self.args.gradient_accumulation_steps
                accumulate_loss+=loss
                self.accelerator.backward(loss)
                accumulate_step+=1
                
                if accumulate_step%self.args.gradient_accumulation_steps==0:
                    self.accelerator.clip_grad_norm_(self.model.parameters(),self.args.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True) # @QUESTION: set_to_none?
                    self.scheduler.step()
                    if completed_steps%self.args.logging_steps==0:
                        accumulate_loss=accumulate_loss.detach()
                        if self.accelerator.state.num_processes>1:
                            accumulate_loss=torch.mean(self.accelerator.gather_for_metrics(accumulate_loss)) # @QUESTION: gather_for_metrics?
                        log={
                            'train_loss':accumulate_loss,
                            'learning_rate':self.scheduler.get_last_lr()[0]
                        }
                        logging_string=f'[{completed_steps}/{self.args.num_training_steps}] Training Loss: {accumulate_loss}'
                        if self.args.log_wandb:
                            self.accelerator.log(log,step=completed_steps)
                    
                    ### Evaluation Loop ###
                    if completed_steps % self.args.evaluation_interval == 0:
                        if self.accelerator.is_main_process:
                            progress_bar.write("Evaluating Model...")
                        
                        self.model.eval()

                        ### Dictionary to Store Results ###
                        log = {"val_loss": 0}

                        ### Iterate Data ###
                        num_losses = 0
                        for batch in tqdm(self.eval_dataloader, disable=not self.accelerator.is_main_process):
                            
                            ### Grab Input IDs ###
                            input_ids = batch["input_ids"].to(self.accelerator.device)

                            ### Attend to All Tokens (EVEN EOS) ###
                            batch_size, seq_len = input_ids.shape
                            attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long, device=self.accelerator.device)

                            ### Random sample t to mask each token with that probability ###'
                            batch_size, seq_len = input_ids.shape
                            t = torch.rand(batch_size, 1, device=self.accelerator.device).expand(batch_size, seq_len)
                            mask = torch.bernoulli(t).bool()

                            ### Mask Data and Dont Compute Loss for Unmasked Data ###
                            masked_input_ids = input_ids.masked_fill(mask, self.tokenizer.mask_token_id)
                            labels = input_ids.masked_fill(~mask, -100)

                            ### Compute Logits ###
                            with torch.inference_mode():
                                logits = self.model(input_ids=masked_input_ids, attention_mask=attention_mask)["logits"]
                            
                            ### Compute Loss (per token) ###
                            num_classes = logits.shape[-1]
                            loss = self.loss_fn(logits.reshape(batch_size*seq_len, num_classes),
                                            labels.flatten())
                            
                            ### Scale loss by t. As t gets larger, we have more mask tokens and its becomes a tougher problem ###
                            ### So naturally samples with large t, will have a worse loss. Just to make it fair we scale our ###
                            ### loss per sample by the t ###
                            loss = loss.reshape(batch_size, seq_len) / t
                            loss = loss.mean()

                            ### Grab Loss ###
                            loss = loss.detach()
                            if self.accelerator.num_processes > 1:
                                loss = torch.mean(self.accelerator.gather_for_metrics(loss))

                            ### Add to our Logs ###
                            log["val_loss"] += loss
                            num_losses += 1
                        
                        ### Divide Log by Num Losses ###
                        log["val_loss"] = log["val_loss"] / num_losses

                        ## Print to Console ###
                        logging_string = f"[{completed_steps}/{self.args.num_training_steps}] Validation Loss: {log['val_loss']}"
                
                        ### Print out Log ###
                        if self.accelerator.is_main_process:
                            progress_bar.write(logging_string)
                        
                        if self.args.log_wandb:
                            self.accelerator.log(log, step=completed_steps)

                        self.model.train()
                    
                    ### Checkpoint Model (Only need main process for this) ###
                    '''if (completed_steps % self.args.checkpoint_interval == 0):
                        
                        ### Save Checkpoint ### 
                        path_to_checkpoint = os.path.join(self.experiment_dir, f"checkpoint_{completed_steps}")

                        if self.accelerator.is_main_process:
                            progress_bar.write(f"Saving Checkpoint to {path_to_checkpoint}")

                        ### Make sure that all processes have caught up before saving checkpoint! ###
                        self.accelerator.wait_for_everyone()

                        ### Save checkpoint using only the main process ###
                        if self.accelerator.is_main_process:
                            self.accelerator.save_state(output_dir=path_to_checkpoint)'''
                    
                    if completed_steps >= self.args.num_training_steps:
                        train = False
                        if self.accelerator.is_main_process:
                            progress_bar.write("Completed Training!!")
                        break

                    ### Iterate Progress Bar and Completed Steps ###
                    completed_steps += 1
                    progress_bar.update(1)

                    ### Reset Loss Accumulate For Next Accumulation ###
                    accumulate_loss = 0

        checkpoint_dir = os.path.join(self.experiment_dir, f"final_model")
        self.accelerator.save_state(output_dir=checkpoint_dir)
        self.accelerator.end_training()
    def sfttrain(self):
        train=True
        completed_steps=0
        progress_bar=tqdm(range(completed_steps,self.args.num_training_steps),disable=not self.accelerator.is_local_main_process) # Since there are multiple processes, we need to display only one process
        while train:
            accumulate_step=0
            accumulate_loss=0
            for batch in self.train_dataloader:
                input_ids=batch['input_ids'].to(self.accelerator.device)
                query_mask=batch['query_mask'].to(self.accelerator.device)
                
                # Let all tokens (even eos) to be masked
                batch_size,seq_len=input_ids.shape
                attention_mask=torch.ones((batch_size,seq_len),dtype=torch.long,device=self.accelerator.device)
                
                # Random sample t to mask tokens with probabilities
                t=torch.rand(batch_size,1,device=self.accelerator.device).expand(batch_size,seq_len).clamp_min(1e-5) # @TODO: re-write this in for loop
                mask=torch.bernoulli(t)
                
                # @QUESTION
                # Mask only answer, not the prompt
                mask=mask*query_mask
                mask=mask.bool()
                
                masked_input_ids=input_ids.masked_fill(mask,self.tokenizer.mask_token_id) # @QUESTION: what is self.tokenizer.mask_token_id?
                labels=input_ids.masked_fill(~mask,-100) # @QUESTION: why -100? what does this mean?
                logits=self.model(input_ids=masked_input_ids,attention_mask=attention_mask)['logits'] # @QUESTION
                num_classes=logits.shape[-1] # @QUESTION
                loss=self.loss_fn(logits.reshape(batch_size*seq_len,num_classes),
                                  labels.flatten())
                
                '''As t gets larger, we will have more mask tokens, and demask these tokens will become a harder problem.
                So for samples with larger t, will have worse loss.
                So to make it fair, we need to scale our losses by deviding them by t.'''
                loss=loss.reshape(batch_size,seq_len)/t # Devide by t in order to scale the loss
                
                # Different answers have different lengths, so we need to scale them
                answer_length=query_mask.sum(dim=1,keepdim=True) # Shape is (batch_size, 1)
                answer_length=answer_length.clamp_min(1) # Just keep data point with length larger or equal to 1
                loss=loss/answer_length
                
                
                # @QUESTION: per-token loss? I think it already was per sentence loss?
                loss=loss.sum(dim=1).mean() # Add up all per-token losses and average them across batch
                
                # QUESTION: why do we need to scale the loss by this?
                # Scale loss by gradient accumualtion steps
                loss=loss/self.args.gradient_accumulation_steps
                accumulate_loss+=loss
                self.accelerator.backward(loss)
                accumulate_step+=1
                if accumulate_step%self.args.gradient_accumulation_steps==0:
                    # Update the model
                    self.accelerator.clip_grad_norm_(self.model.parameters(),self.args.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    
                    # Log results
                    if completed_steps%self.args.logging_steps==0:
                        accumulate_loss=accumulate_loss.detach()
                        
                        # @QUESTION: why do this
                        if self.accelerator.state.num_processes>1:
                            accumulate_loss=torch.mean(self.accelerator.gather_for_metrics(accumulate_loss)) # @QUESTION: what is gather_for_metrics?
                        
                        log={
                            'train_loss':accumulate_loss,
                            'learning_rate':self.scheduler.get_last_lr()[0]
                        }
                        logging_string=f'[{completed_steps}/{self.args.num_training_steps}] Training Loss: {accumulate_loss}'
                        if self.accelerator.is_main_process:
                            progress_bar.write(logging_string)
                        if self.args.log_wandb:
                            self.accelerator.log(log,step=completed_steps)
                    
                    # Evaluation 
                    if completed_steps%self.args.evaluation_interval==0:
                        if self.accelerator.is_main_process:
                            progress_bar.write('Evaluating Model...')
                        self.model.eval()
                        log={'val_loss':0}
                        num_losses=0
                        for batch in tqdm(self.eval_dataloader,disable=not self.accelerator.is_main_process):
                            input_ids=batch['input_ids'].to(self.accelerator.device)
                            query_mask=batch['query_mask'].to(self.accelerator.device)
                            batch_size,seq_len=input_ids.shape
                            attention_mask=torch.ones((batch_size,seq_len),dtype=torch.long,device=self.accelerator.device)
                            t=torch.rand(batch_size,1,device=self.accelerator.device).expand(batch_size,seq_len)
                            mask=torch.bernoulli(t).bool()
                            mask=mask*query_mask
                            mask=mask.bool()
                            masked_input_ids=input_ids.masked_fill(mask,self.tokenizer.mask_token_id)
                            labels=input_ids.masked_fill(~mask,-100)
                            with torch.inference_mode(): # @STAR: new stuff
                                logits=self.model(input_ids=masked_input_ids,attention_mask=attention_mask)['logits']
                            num_classes=logits.shape[-1]
                            loss=self.loss_fn(logits.reshape(batch_size*seq_len,num_classes),
                                              labels.flatten())
                            loss=loss.reshape(batch_size,seq_len)/t
                            answer_length=query_mask.sum(dim=1,keepdim=True)
                            answer_length=answer_length.clamp_min(1)
                            loss=loss/answer_length
                            loss=loss.sum(dim=1).mean()
                            loss=loss.detach()
                            if self.accelerator.num_processes>1:
                                loss=torch.mean(self.accelerator.gather_for_metrics(loss))
                            log['val_loss']+=loss
                            num_losses+=1
                        log['val_loss']=log['val_loss']/num_losses
                        logging_string=f'[{completed_steps}/{self.args.num_training_steps}] Validation Loss: {log["val_loss"]}'
                        if self.accelerator.is_main_process:
                            progress_bar.write(logging_string)
                        if self.args.log_wandb:
                            self.accelerator.log(log,step=completed_steps)
                        self.model.train()
                    
                    # Save checkpoint
                    '''if completed_steps%self.args.checkpoint_interval==0:
                        checkpoint_dir=os.path.join(self.experiment_dir,f'checkpoint_{completed_steps}')
                        if self.accelerator.is_main_process:
                            progress_bar.write(f'Saving checkpoint to: {checkpoint_dir}')
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            self.accelerator.save_state(output_dir=checkpoint_dir)'''
                    
                    if completed_steps>=self.args.num_training_steps:
                        train=False
                        if self.accelerator.is_main_process:
                            progress_bar.write('Completed Training')
                        break
                    
                    # Update progess bar and reset this accumulation
                    completed_steps+=1
                    progress_bar.update(1)
                    accumulate_loss=0
        checkpoint_dir=os.path.join(self.experiment_dir,'final_model')
        self.accelerator.save_state(output_dir=checkpoint_dir)
        self.accelerator.end_training()