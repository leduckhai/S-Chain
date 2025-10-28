#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM, \
                         CLIPVisionModel, CLIPImageProcessor

from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from llava.model.utils import *
from llava.model.qformer import init_tokenizer, init_Qformer
from llava.model.dense_connector import dense_connector
import open_clip
import os, json
import re

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}
    
def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    if projector_type == 'linear':
        print("--------------------------This is version 1.0---------------------")
        print(projector_type)
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        print("--------------------------This is version 1.5---------------------")
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)
    if projector_type == 'qformer': 
        print("--------------------------Using projection from Qformer---------------------")
        return nn.Linear(config.qformer_hidden_size, config.hidden_size)
    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def initialize_qformer(config): 
        qformer_tokenizer = init_tokenizer(truncation_side="left")
        ln_vision = nn.LayerNorm(config.mm_hidden_size) # num_features of ViT-L/14 from CLIP
        Qformer, query_tokens = init_Qformer(32, config.mm_hidden_size)
        Qformer.resize_token_embeddings(len(qformer_tokenizer))
        Qformer.cls = None

        state_dict = torch.load(config.qformer_path, map_location="cpu")['model']
        Qformer_state_dict = OrderedDict()
        ln_vision_state_dict = OrderedDict()

        for key in state_dict.keys(): 
            if 'Qformer' in key: 
                Qformer_state_dict[key.replace('Qformer.', '')] = state_dict[key]
            if 'ln_vision' in key: 
                ln_vision_state_dict[key.replace('ln_vision.', '')] = state_dict[key]

        query_tokens.data.copy_(state_dict['query_tokens'])

        config.qformer_hidden_size = Qformer.config.hidden_size

        return (qformer_tokenizer, ln_vision, query_tokens, Qformer)

class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig, mm_vision_tower=None, mm_hidden_size=None):
        super(LlavaLlamaModel, self).__init__(config) # config is taken from file /netscratch/trnguyen/llava_med_checkpoints_llama/LLaVA-7b-v0/config.json

        self.vision_tower_name = "openai/clip-vit-large-patch14" # microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 # openai/clip-vit-large-patch14
        if hasattr(config, "mm_vision_tower"):
            # HACK: for FSDP
            if "BiomedCLIP" in config.mm_vision_tower or "biomed_clip" in config.mm_vision_tower:
                model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
                self.vision_tower = [model.visual.trunk] # Please refer: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/timm_model.py#LL60C18-L60C18
                self.vision_tower_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            else:
                self.vision_tower = [CLIPVisionModel.from_pretrained(config.mm_vision_tower)]


        #if hasattr(config, "use_mm_proj"):
        if hasattr(config, "mm_projector_type"):
            print("--------Build mm_projector during model initialization--------")
            # self.mm_projector = nn.Linear(config.mm_hidden_size, config.hidden_size)
            if config.mm_projector_type == 'qformer': # have to make sure that config.qformer_path exists if mm_projector_type = 'qformer'
                self.qformer_tokenizer, self.ln_vision, self.query_tokens, self.Qformer = initialize_qformer(config)
            self.mm_projector = build_vision_projector(config)

    def initialize_vision_modules(self, model_args, vision_tower, mm_vision_select_layer,
                                  pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):

        #############
        self.contrastive = getattr(model_args, 'contrastive', False)
        self.mm_dense_connector_type = model_args.mm_dense_connector_type
        self.num_l = model_args.num_l
        if self.contrastive:
            self.alpha = getattr(model_args, 'alpha', 1.0)
            self.contrastive_loss_type = getattr(model_args, 'contrastive_loss_type', "infonce")
            # self.temperature = getattr(model_args, 'temperature', 100.0)
            self.temperature = nn.Parameter(data=torch.tensor(4.606), requires_grad=True)
            if self.contrastive_loss_type == "siginfo":
                self.beta = nn.Parameter(data=torch.tensor(10), requires_grad=True)
            elif self.contrastive_loss_type == "twins": 
                self.lambd = getattr(model_args, 'lambd', 0.0051)
                self.bn = nn.BatchNorm1d(self.config.hidden_size, affine=False)
        #############
        if "BiomedCLIP" in vision_tower:
            self.vision_tower_name = "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
            return self.initialize_vision_modules_from_biomed_clip(model_args, vision_tower, mm_vision_select_layer,
                                pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False)
        else:
            return self.initialize_vision_modules_from_openai_clip(model_args, vision_tower, mm_vision_select_layer,
                                pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False)



    def initialize_vision_modules_from_openai_clip(self, model_args, vision_tower, mm_vision_select_layer,
                                  pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
        self.config.mm_vision_tower = vision_tower #model_args (arguement added via command line) is added to configuration in config.json

        image_processor = CLIPImageProcessor.from_pretrained(vision_tower)

        if not hasattr(self, 'vision_tower'):
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower)
        else:
            vision_tower = self.vision_tower[0]
        vision_tower.requires_grad_(False)
        vision_tower = vision_tower.to(torch.float16)
        self.vision_tower = [vision_tower]

        vision_config = vision_tower.config
        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.qformer_path = getattr(model_args, 'qformer_path', '/netscratch/trnguyen/instructBLIP_checkpoint/blip2_pretrained_vitL.pth')
        if self.mm_dense_connector_type in ['dci', 'sci']:
            self.config.mm_hidden_size = vision_config.hidden_size*3
        else:
            self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer

        if not hasattr(self, 'mm_projector'):
            # self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)
            if self.config.mm_projector_type == 'qformer':
                 self.qformer_tokenizer, self.ln_vision, self.query_tokens, self.Qformer = initialize_qformer(self.config)
            self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        return dict(
            image_processor=image_processor,
            image_token_len=num_patches,
            vision_config=vision_config
        )

    def initialize_vision_modules_from_biomed_clip(self, model_args, vision_tower, mm_vision_select_layer,
                                  pretrain_mm_mlp_adapter=None, tune_mm_mlp_adapter=False):
        self.config.mm_vision_tower = vision_tower

        

        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
        openai_vision_tower = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch16")
        vision_config = openai_vision_tower.config
        del openai_vision_tower
                
        if not hasattr(self, 'vision_tower'):
            model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
            vision_tower = model.visual.trunk # Please refer: https://github.com/mlfoundations/open_clip/blob/main/src/open_clip/timm_model.py#LL60C18-L60C18

            # from huggingface_hub import snapshot_download
            # BiomedCLIP_file_path = "biomed-clip-share"
            # # snapshot_download("microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224", local_dir=BiomedCLIP_file_path)
            # with open(os.path.join(BiomedCLIP_file_path, "open_clip_config.json"), 'r') as file:  
            #     config = json.load(file) 


        else:
            vision_tower = self.vision_tower[0]

        
        setattr(vision_tower, 'config', vision_config)
        vision_tower.requires_grad_(False)
        vision_tower = vision_tower.to(torch.float16)
        self.vision_tower = [vision_tower]

        num_patches = (vision_config.image_size // vision_config.patch_size) ** 2

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        if self.mm_dense_connector_type in ['dci', 'sci']:
            self.config.mm_hidden_size = vision_config.hidden_size*3
        else:
            self.config.mm_hidden_size = vision_config.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer

        if not hasattr(self, 'mm_projector'):
            # self.mm_projector = nn.Linear(vision_config.hidden_size, self.config.hidden_size)
            self.mm_projector = build_vision_projector(self.config)

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            self.mm_projector.load_state_dict({k.split('.')[-1]: v for k, v in mm_projector_weights.items()})

        return dict(
            image_processor=image_processor,
            image_token_len=num_patches,
            vision_config=vision_config
        )

    def extract_visual_features(self, vision_tower, images):
        select_hidden_state_layer = getattr(self.config, "mm_vision_select_layer", -1)

        
        if "BiomedCLIP" in self.vision_tower_name  or "biomed_clip" in self.vision_tower_name:
            image_forward_outs = vision_tower.get_intermediate_layers(images, n=25) # take last n blocks if n is an int, if in is a sequence, select by matching indices
            print("======================len BIOMED========================")
            print(len(image_forward_outs))
            image_features = image_forward_outs[select_hidden_state_layer]
            image_features = image_features
            if self.mm_dense_connector_type in ['sti', 'sci', 'dci']:
                print("===========use DCI=============")
                image_features = dense_connector(image_features=image_features, image_forward_outs=image_forward_outs,
                                                 mm_dense_connector_type=self.mm_dense_connector_type, is_biomed=True, num_l=self.num_l)
        else:
            image_forward_outs = vision_tower(images, output_hidden_states=True)
            select_hidden_state = image_forward_outs.hidden_states[select_hidden_state_layer]
            if self.config.mm_projector_type == 'qformer': 
                image_features = select_hidden_state # [B, 257, 1024] with [CLS] at first position
            else: 
                image_features = select_hidden_state[:, 1:] # [B, 256, 1024], 256 here is number of patch
            if self.mm_dense_connector_type in ['sti', 'sci', 'dci']:
                # print("===========use DCI=============")
                image_features = dense_connector(image_features=image_features, image_forward_outs=image_forward_outs,
                                                 mm_dense_connector_type=self.mm_dense_connector_type, is_biomed=False, num_l=self.num_l)
        return image_features
    def off_diagonal(self,x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        text_input: Optional[List[str]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        #input_ids: tensor size [B, T] with T is length of longest sentence, B is batch size
        #attention_mask: same size as input_ids
        #images: [B,3,224,224]
        #return_dict: True, output_attentions: False, output_hidden_states: False
        #other is None
        # HACK: replace back original embeddings for LLaVA pretraining
        orig_embeds_params = getattr(self, 'orig_embeds_params', None)
        # if orig_embeds_params is not None:
        #     orig_embeds_params = orig_embeds_params[0]
        #     with torch.no_grad():
        #         self.get_input_embeddings().weight.data[:-2] = orig_embeds_params[:-2].data

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) # [B, T, hidden_size] e.g [4,329,4096], E in CG-VLM

        vision_tower = getattr(self, 'vision_tower', None)
        lossAlign = 0
        if vision_tower is not None and (input_ids.shape[1] != 1 or self.training) and images is not None:
            # TODO: this is a modified multimodal LLM -- Haotian Liu
            vision_tower = vision_tower[0]  # HACK: for FSDP
            with torch.no_grad():
                if type(images) is list:
                    # variable length images
                    image_features = []
                    for image in images:
                        image_feature = self.extract_visual_features(vision_tower, image.unsqueeze(0))
                        image_features.append(image_feature)
                else:
                    image_features = self.extract_visual_features(vision_tower, images) # vision_tower is frozen, image_features: [B, num_patch, vision_config.hidden_size] = [4,256, 1024], V in CG-VLM
            ############################### Add Qformer here ##########################################
            # We need to do the following steps
            #1. Create a new data format for input_ids 
            #   1.1. (number of <im_patch> is now 32 instead of 256)
            #   1.2. text_input (system message + human's question) needs to be in text format (to feed in Qformer's tokenizer)
            #   1.3. need to make sure that loss is not applied to to the query tokens ( aka <im_patch>) (pretty sure loss is not applied according to paper LLaVA)
            #2. Tokenize the text_input using Qformer's tokenizer
            #3. Pass the tokenized text_input, query_tokens and image_features (which is self.ln_vision(self.extract_visual_features(...))) to Qformer.bert() to get query_output
            #4. Pass query_output through self.mm_projector to get inputs_llm (= image_features = self.mm_projector(image_features))
            #5. Concatenate (interleave) embedded, tokenized conversations (question +  answer) with inputs_llm (this is exactly the same as original LLaVA's code so we don't do anything). The final result is inputs_embeds
            #6. Pass inputs_embeds to super(LlavaLlamaModel, self).forward()
            # => so we need to get query_output here (before calling self.mm_projector)
            #Need to make sure that Qformer and other relevant components are trainable (requires_grad = True)
            
            if self.config.mm_projector_type == 'qformer':
                image_embeds = self.ln_vision(image_features) # encode the image [B, 257, 1024]
                image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(images.device) # [B, 257]
                query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1) # [B, 32, 768]
                text_Qformer = self.qformer_tokenizer(
                    text_input,
                    padding='longest',
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(images.device) # tokenized the text prompt as input to Qformer. Qformer.input_ids: tensor size [B, T]
                query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(images.device) # [B, 32]
                Qformer_atts = torch.cat([query_atts, text_Qformer.attention_mask],dim=1)

                query_output = self.Qformer.bert(
                    text_Qformer.input_ids,
                    attention_mask=Qformer_atts,
                    query_embeds=query_tokens,
                    encoder_hidden_states=image_embeds,
                    encoder_attention_mask=image_atts,
                    return_dict=True,
                    )

                image_features = query_output.last_hidden_state[:,:query_tokens.size(1),:] # query_output.last_hidden_state: [batch, num_query + n_token, hidden_size_Bert]

            ###########################################################################################
            if type(images) is list:
                image_features = [self.mm_projector(image_feature)[0] for image_feature in image_features]
            else:
                image_features = self.mm_projector(image_features) # [B, num_patch, config.hidden_size] = [4, 256, 4096], Z in CG-VLM or [B, num_query, config.hidden_size] = [4,32,4096] if using Qformer
            
            
            ##########
            if self.contrastive:
                align_input_embeds = []
            ##########
            new_input_embeds = []
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, inputs_embeds): #loop through each conversation in a batch, cur_input_embeds: [329, 4096]
                if (cur_input_ids == vision_tower.config.im_patch_token).sum() == 0:
                    # multimodal LLM, but the current sample is not multimodal
                    new_input_embeds.append(cur_input_embeds)
                    continue
                if vision_tower.config.use_im_start_end:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_start_token).sum() != (cur_input_ids == vision_tower.config.im_end_token).sum():
                        raise ValueError("The number of image start tokens and image end tokens should be the same.")
                    image_start_tokens = torch.where(cur_input_ids == vision_tower.config.im_start_token)[0]
                    if self.contrastive: 
                        image_end_tokens = torch.where(cur_input_ids == vision_tower.config.im_end_token)[0]
                        pad_start_tokens = torch.where(cur_input_ids == 32000)[0]
                        if pad_start_tokens.nelement() == 0: 
                            pad_start_tokens = torch.tensor([len(cur_input_ids)]).to(cur_input_ids.device)
                        else: 
                            pad_start_tokens = pad_start_tokens[0].reshape(1)
                    for image_start_token_pos in image_start_tokens:
                        cur_image_features = image_features[cur_image_idx].to(device=cur_input_embeds.device)
                        num_patches = cur_image_features.shape[0]

                        # import pdb; pdb.set_trace()
                        if cur_input_ids[image_start_token_pos + num_patches + 1] != vision_tower.config.im_end_token:
                            raise ValueError("The image end token should follow the image start token.")
                        if orig_embeds_params is not None:
                            # concat embedding of text with embedding of image, detach() text and keep image (<im_start><im_patch>...<im_end)
                            #Looks like we have to change the format of input dataset (i.e instead of 256 patch, only 32 patch)
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos].detach(), cur_input_embeds[image_start_token_pos:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:image_start_token_pos + num_patches + 2], cur_input_embeds[image_start_token_pos + num_patches + 2:].detach()), dim=0)
                        else:
                            cur_new_input_embeds = torch.cat((cur_input_embeds[:image_start_token_pos+1], cur_image_features, cur_input_embeds[image_start_token_pos + num_patches + 1:]), dim=0)
                        cur_image_idx += 1
                    new_input_embeds.append(cur_new_input_embeds)
                    ##############
                    if self.contrastive:
                        for image_end_token_pos, pad_start_tokens_pos in zip(image_end_tokens, pad_start_tokens):
                            cur_new_input_embeds = cur_input_embeds[image_end_token_pos+5:pad_start_tokens_pos-4]
                        if self.contrastive_loss_type != "twins":
                            align_input_embeds.append(cur_new_input_embeds)# list of [T_i, 4096], with T_i is length of sentence i in batch 
                        else: 
                            align_input_embeds.append(cur_new_input_embeds.mean(dim=0))

                    ##############
                else:
                    cur_image_features = image_features[cur_image_idx]
                    num_patches = cur_image_features.shape[0]
                    if (cur_input_ids == vision_tower.config.im_patch_token).sum() != num_patches:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                    masked_indices = torch.where(cur_input_ids == vision_tower.config.im_patch_token)[0]
                    mask_index_start = masked_indices[0]
                    if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patches, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                        raise ValueError("The image patch tokens should be consecutive.")
                    if orig_embeds_params is not None:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start].detach(), cur_image_features, cur_input_embeds[mask_index_start+num_patches:].detach()), dim=0)
                    else:
                        cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patches:]), dim=0)
                    new_input_embeds.append(cur_new_input_embeds)
            inputs_embeds = torch.stack(new_input_embeds, dim=0) # [4,329,4096] ?
            if self.contrastive and self.contrastive_loss_type == "twins": 
                align_input_embeds = torch.stack(align_input_embeds, dim= 0) # [4,4096] ?
            #####################
            if self.contrastive:
                # print("Use contrastive mechanism.")
                align_image_features = image_features.clone() # [B, num_patch, config.hidden_size] = [4, 256, 4096], Z in CG-VLM
                align_image_features_pool = align_image_features.mean(dim = 1) # [B, config.hidden_size] = [4,4096]
                
                if self.contrastive_loss_type == "infonce" or self.contrastive_loss_type == "siginfo":
                    align_image_features_pool = align_image_features_pool/align_image_features_pool.norm(dim=1)[:, None]

                    matrixSimi = torch.zeros(align_image_features.shape[0], align_image_features.shape[0], device=image_features.device, dtype=image_features.dtype)
                    for i, align_input in enumerate(align_input_embeds): # align_input: [T_j, 4096], E^j
                        align_input_norm = align_input/align_input.norm(dim=1)[:, None]
                        # simi = F.cosine_similarity(align_image_features_pool[None, :, :], align_input[:,None,:], dim=-1).T
                        simi = torch.mm(align_image_features_pool, align_input_norm.transpose(0, 1))

                        simi = simi.mean(dim = 1)
                        matrixSimi[:, i] = torch.exp(self.temperature)*simi
                
                    if self.contrastive_loss_type == "infonce":
                        targetAlign = torch.arange(align_image_features.shape[0]).to(image_features.device)
                        lossAlign = CrossEntropyLoss()(matrixSimi, targetAlign)
                        lossAlign = lossAlign*self.alpha
                    elif self.contrastive_loss_type == "siginfo":
                        matrixSimi += self.beta
                        targetAlign = (2*torch.eye(align_image_features.shape[0]) - torch.ones(align_image_features.shape[0])).to(image_features.device)
                        lossAlign = -1.0*torch.sum(torch.nn.LogSigmoid()(matrixSimi*targetAlign))/align_image_features.shape[0]
                        lossAlign = lossAlign*self.alpha
                elif self.contrastive_loss_type == "twins": # TODO: fix twins by duplicating one image to the number of tokens in the corresponding sequence and applying eq (3) in CG-VLM
                    c = self.bn(align_image_features_pool).T @ self.bn(align_input_embeds)
                    c.div_(align_image_features_pool.shape[0]*torch.cuda.device_count())
                    torch.distributed.all_reduce(c)
                    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
                    off_diag = self.off_diagonal(c).pow_(2).sum()
                    lossAlign = on_diag + self.lambd * off_diag
                    lossAlign = lossAlign*self.alpha
                    
            #####################
        if not hasattr(self, 'alpha'):
            self.alpha = 1.0

        if not hasattr(self, 'temperature'):
            self.temperature = torch.tensor(0)
            
        return super(LlavaLlamaModel, self).forward(
            input_ids=None, attention_mask=attention_mask, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache,
            output_attentions=output_attentions, output_hidden_states=output_hidden_states,
            return_dict=return_dict
        ), lossAlign, self.alpha, torch.exp(self.temperature)


class LlavaLlamaForCausalLM(LlamaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config): # config is taken from file /netscratch/trnguyen/llava_med_checkpoints_llama/LLaVA-7b-v0/config.json
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        text_input: Optional[List[str]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs, lossAlign, alphaArg, tempPa = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            text_input = text_input,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            images=images
        )

        hidden_states = outputs[0] # size [B, T, config.hidden_size] = (e.g) [4,329, 4096], T is length of longest sequence in the batch
        logits = self.lm_head(hidden_states) # [B, T, vocab_size], e.g [4, 329, 32004]

        loss = None
        if labels is not None:
            # Shift so that tokens before position n predict n (tokens < n predict n)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        if loss is not None: 
            loss = loss + lossAlign
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "text_input": kwargs.get("text_input", None),
            }
        )
        return model_inputs

    def initialize_vision_tokenizer(self, mm_use_im_start_end, tokenizer, device,
                                    tune_mm_mlp_adapter=False, pretrain_mm_mlp_adapter=None):
        vision_config = self.model.vision_tower[0].config
        vision_config.use_im_start_end = mm_use_im_start_end
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.resize_token_embeddings(len(tokenizer))

        if mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if tune_mm_mlp_adapter:
                self.model.orig_embeds_params = [self.get_input_embeddings().weight.data.clone().to(device=device)]
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")

        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)