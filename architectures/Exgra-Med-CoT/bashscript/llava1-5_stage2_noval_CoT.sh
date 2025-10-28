export WANDB_API_KEY=b40bed659a04b35785b0534935b459f944bdf6e8

lr=2e-5
epochs=3
batchsize=16
prompt_mode=cot
version=_exgra_med_on_CoT_dataset_epochs${epochs}_batchsize${batchsize}_lr${lr}_prompt_mode_${prompt_mode}
# model_name_or_path=/netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LVLM-Med/models/checkpoint_exgra_med_on_CoT_dataset_epochs3_batchsize16_reasoning
model_name_or_path=/netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LVLM-Med/models/checkpoint_llava_med_instruct_60k_inline_mention_version_1-5_1e0_multi_graph_100_scale_test_bugfix
output_dir=/netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LVLM-Med/weights_finetuned/CoT-100${version}
run_name=CoT-100${version}
answers_file=/netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LVLM-Med/test_answer/CoT-100${version}.jsonl

torchrun --nnodes=1 --nproc_per_node=2 --master_port=25057 \
    llava/train/train_mem_CoT.py \
    --model_name_or_path=${model_name_or_path} \
    --data_path /netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LVLM-Med/data/llava_med_mri_bbox_train_CoT.json \
    --image_folder /netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LVLM-Med/data/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --mm_dense_connector_type none \
    --num_l 6 \
    --prompt_mode=${prompt_mode} \
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
    # --run_name=${run_name}

# sleep 5s 
#change the following
# --model-name: same as --output_dir above, which is path to load the model
# --answers-file: file to store the result
python llava/eval/run_med_datasets_eval_batch_CoT.py \
    --num-chunks 2 \
    --conv-mode ${prompt_mode} \
    --model-name ${output_dir} \
    --mm_dense_connector_type none \
    --num_l 6 \
    --question-file /netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LVLM-Med/data/llava_med_mri_bbox_test_CoT.json \
    --image-folder /netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LVLM-Med/data/images \
    --answers-file ${answers_file}

# change the following
# --pred: same as --answers-file above
# the metrics (recall and accuracy) are saved as a text file in the same place, with the same name as --pred. 
# E.g: if --pred is ans-opt-new-3.jsonl, then metrics are saved in ans-opt-new-3.txt
python llava/eval/run_eval_CoT.py \
    --gt /netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LVLM-Med/data/llava_med_mri_bbox_test_CoT.json \
    --pred ${answers_file} \
    --candidate /netscratch/duynguyen/Research/Slake1.0/candidate.json
