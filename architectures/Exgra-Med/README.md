<div align='center'>

## Exgra-Med and LLaVA-Med on CoT Dataset

</div>

### Usage

#### Exgra-Med
To run the model settings on our dataset, check running files in folder bashscript. Then, create folder ``./checkpoints`` and download the Exgra-Med weight from [link](https://exgra-med.github.io/) and put into ``./checkpoints`` folder before running.

#### LLaVA-Med
Because the Exgra-Med and LLaVA-Med use the same codebase for finetuning, you can change variables ``model_name_or_path`` (for original weight) and ``version`` (for version running) in running ``.sh`` scripts to finetune LLaVA-Med on this dataset. The checkpoint for LLaVA-Med can be also downloaded in this [link](https://exgra-med.github.io/).

For example, to run LLaVA-Med on our Structured Visual CoT, where each step links to image regions (SV-CoT), change the scripts ``sv_cot.sh`` into:

```Shell
# This script is used to train model with input image, question, our Structured Visual CoT, and final prediction (Q4).
export WANDB_API_KEY= #<your_wandb_key>

lr=2e-5
epochs=3
batchsize=16
prompt_mode=cot
use_rag=false
version=_llava_med_on_CoT_dataset_epochs${epochs}_batchsize${batchsize}_lr${lr}_prompt_mode_${prompt_mode}_useRAG${use_rag}
model_name_or_path=./checkpoints/llava-med
output_dir=./checkpoints/CoT-100${version}
run_name=CoT-100${version}
answers_file=./test_answer/CoT-100${version}.jsonl

train_file_json=./data/llava_med_mri_bbox_train_CoT.json
test_file_json=./data/llava_med_mri_bbox_test_CoT.json
image_folder=./data/images

torchrun --nnodes=1 --nproc_per_node=2 --master_port=25057 \
    llava/train/train_mem_CoT.py \
    --model_name_or_path=${model_name_or_path} \
    --data_path=${train_file_json} \
    --image_folder=${image_folder} \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --mm_dense_connector_type none \
    --num_l 6 \
    --prompt_mode=${prompt_mode} \
    --use_rag=${use_rag} \
    --bf16 True \
    --output_dir=${output_dir} \
    --num_train_epochs=${epochs} \
    --per_device_train_batch_size=${batchsize} \
     --per_device_eval_batch_size 64 \
    --gradient_accumulation_steps 1 \
    --save_strategy "steps" \
     --save_steps 101100 \
     --save_total_limit 4 \
    --learning_rate=${lr} \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --fsdp "full_shard auto_wrap" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --lazy_preprocess True \
    --report_to none \
    --run_name=${run_name}

######### Run Evaluation #########

python llava/eval/run_med_datasets_eval_batch_CoT.py \
    --num-chunks 2 \
    --conv-mode ${prompt_mode} \
    --use_rag ${use_rag} \
    --model-name ${output_dir} \
    --mm_dense_connector_type none \
    --num_l 6 \
    --question-file ${test_file_json} \
    --image-folder ${image_folder} \
    --answers-file ${answers_file}

python llava/eval/run_eval_CoT.py \
    --gt ${test_file_json} \
    --pred ${answers_file} \
```

#### Hyperparameters Settings
In these files in ``bashscript`` folder, we can change these settings.
```Shell
export WANDB_API_KEY= #<your_wandb_key>

lr=2e-5
epochs=3
batchsize=16
prompt_mode=simple # cot for running CoT mode or simple for running original mode
use_rag=true # True if use RAG to change to RAG system prompt, else False
version=_exgra_med_on_No_CoT_dataset_epochs${epochs}_batchsize${batchsize}_lr${lr}_prompt_mode_${prompt_mode}_useRAG${use_rag}
model_name_or_path=./checkpoints/exgra-med # Path for original Exgra-Med weight
output_dir=./checkpoints/CoT-100${version} # Output checkpoint
run_name=CoT-100${version}
answers_file=./test_answer/CoT-100${version}.jsonl # Output predicted answer file

train_file_json=./data/llava_med_mri_bbox_train_CoT.json # Training dataset path
test_file_json=./data/llava_med_mri_bbox_test_CoT.json # Testing dataset path
image_folder=./data/images # Image folder
```
