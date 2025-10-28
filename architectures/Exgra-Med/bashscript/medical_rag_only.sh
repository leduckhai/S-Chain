# This script is used to train model with input image, question, and medical Retrieval-Augmented Generation context
export WANDB_API_KEY= #<your_wandb_key>

lr=2e-5
epochs=3
batchsize=16
prompt_mode=simple
use_rag=true
version=_exgra_med_on_No_CoT_dataset_epochs${epochs}_batchsize${batchsize}_lr${lr}_prompt_mode_${prompt_mode}_useRAG${use_rag}
model_name_or_path=./models/checkpoint_llava_med_instruct_60k_inline_mention_version_1-5_1e0_multi_graph_100_scale_test_bugfix
output_dir=./weights_finetuned/CoT-100${version}
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

# change the following
# --model-name: same as --output_dir above, which is path to load the model
# --answers-file: file to store the result
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

# change the following
# --pred: same as --answers-file above
# the metrics (recall and accuracy) are saved as a text file in the same place, with the same name as --pred. 
# E.g: if --pred is ans-opt-new-3.jsonl, then metrics are saved in ans-opt-new-3.txt
python llava/eval/run_eval_CoT.py \
    --gt ${test_file_json} \
    --pred ${answers_file} \
