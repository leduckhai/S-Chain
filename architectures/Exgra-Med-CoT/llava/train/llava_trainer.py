import os
import torch
import torch.nn as nn

from transformers import Trainer
from typing import Dict, Optional, Union, Any, List, Tuple
from llava.eval.model_vqa_med import KeywordsStoppingCriteria
from tqdm import tqdm
DEFAULT_IM_END_TOKEN = "<im_end>"

def unwrap_model(model: nn.Module) -> nn.Module:
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


class LLaVATrainer(Trainer):

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            # Save the model
            _state_dict = state_dict
            if _state_dict is None:
                # Only save the model itself if we are using distributed training
                model_to_save = unwrap_model(self.model)
                _state_dict = model_to_save.state_dict()

            weight_to_save = {}
            keys_to_match = ['mm_projector', 'embed_tokens', 'embed_in']
            for k, v in _state_dict.items():
                if any(key_match in k for key_match in keys_to_match):
                    weight_to_save[k] = v.cpu().clone().detach() # Chunyuan: to solve the saving OOM problem 

            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))

        super(LLaVATrainer, self)._save(output_dir, state_dict)
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments. Keys:
                'input_ids': torch tensor size [B, T] with B is batch size, T is longest sequence within batch 
                'labels': tensor size [B,T]
                'attention_mask': tensor size [B,T]
                'images': tensor size [B, C, H, W]

            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`List[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        #loss: e.g tensor(0.4545)
        #logits: [B, T, vocab_size] e.g [64, 335, 32004]
        #labels: [B, T] e.g [64, 335]
        loss, _, labels = super(LLaVATrainer, self).prediction_step(model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys)
        #use model.generate() here, do not convert token_ids back to text yet
        #then we return (loss, generated_text_tokens, labels)
        #(generated_text_tokens,labels) will be fed to compute_metrics(), there we convert token_ids back to text
        #we need to change the inputs for validation case a bit, especially when formatting the conversation. (change before or in _add_speaker_and_signal)
        # TODO:  generate text in batch way, currently one sequence at a time (similar to author). Problem: stopping criteria different for each sample in batch. 
        im_end_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_IM_END_TOKEN)
        im_end_token_pos_list = (inputs['input_ids'] == im_end_id).nonzero()[:,1] # tensor of all positions where image tokens end.

        dtype = torch.float32
        if self.args.fp16:
            dtype = torch.float16
        if self.args.bf16:
            dtype = torch.bfloat16
        
        generated_tokens_ids = []
        pbar = tqdm(total = len(im_end_token_pos_list))
        for im_end_token_pos, conversation, image, text_input  in zip(im_end_token_pos_list, inputs['input_ids'], inputs['images'], inputs['text_input']): 
            prompt = conversation[:im_end_token_pos+2].unsqueeze(0).to(self.args.device) # [1,T]
            image = image.unsqueeze(0).to(dtype= dtype, device = self.args.device)
            keywords = ['###']
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, prompt)

            with torch.inference_mode():
                output_ids = model.generate(
                    prompt,
                    text_input=[text_input],
                    images=image,
                    do_sample=False,
                    temperature=0.7,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria], 
                    synced_gpus = True) # [1,T], output_ids also include prompt 
            
            input_token_len = prompt.shape[1]
            n_diff_input_output = (prompt != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] Sample: {n_diff_input_output} output_ids are not the same as the input_ids')
            #outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
            #print(outputs)
            generated_tokens_ids.append(output_ids[:, input_token_len:].squeeze())
            pbar.update(1)
        generated_tokens_ids = torch.nn.utils.rnn.pad_sequence(generated_tokens_ids, batch_first =True, padding_value=self.tokenizer.pad_token_id) # size [B, T1] with T1 is length of longest sequence
        pbar.close()

        return (loss, generated_tokens_ids, labels)