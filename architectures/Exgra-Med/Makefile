build_llava_image:
	clear
	srun -p A100-IML --ntasks 1 --gpus-per-task 1 --cpus-per-gpu=64 --mem-per-cpu 4G --container-mounts=/netscratch/duynguyen:/netscratch/duynguyen --container-image=/netscratch/duynguyen/Research/KOTORI-LLaVA-Med/llava_finetuning.sqsh --container-save=/netscratch/duynguyen/Research/KOTORI-LLaVA-Med/llava_finetuning.sqsh --pty /bin/bash	
.PHONY: build_llava_image

bao_build_llava_image:
	clear
	srun -p A100-IML --ntasks 1 --gpus-per-task 1 --cpus-per-gpu=64 --mem-per-cpu 4G --container-mounts=/netscratch/duynguyen:/netscratch/duynguyen --container-image=/netscratch/duynguyen/Research/bao_llava_med/bao_llava_med.sqsh --container-save=/netscratch/duynguyen/Research/bao_llava_med/bao_llava_med.sqsh --pty /bin/bash	
.PHONY: bao_build_llava_image

download_data_0:
	srun -p A100-IML -t 6-23:59:59 --ntasks 1 \
					--mem-per-cpu 40G\
					--container-image=/netscratch/duynguyen/Research/medllm_new/Docker/medical-llama2-tannp-A100.sqsh \
					--container-workdir="`pwd`" \
					--container-mounts=/netscratch/duynguyen:/netscratch/duynguyen,/netscratch/iml_ssl:/netscratch/iml_ssl,/ds:/ds:ro,"`pwd`":"`pwd`" \
					--export="NCCL_IB_DISABLE=1" \
					--export="OMP_NUM_THREADS=10" \
					--export="LOGLEVEL=INFO" \
					--export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
					--export="FI_PROVIDER='efa'" \
					--export="CUDA_LAUNCH_BLOCKING=1" bash bashscript/download_data_0.sh 
.PHONY: download_data_0

download_data_1:
	srun -p A100-IML -t 6-23:59:59 --ntasks 1 \
					--mem-per-cpu 40G\
					--container-image=/netscratch/duynguyen/Research/medllm_new/Docker/medical-llama2-tannp-A100.sqsh \
					--container-workdir="`pwd`" \
					--container-mounts=/netscratch/duynguyen:/netscratch/duynguyen,/netscratch/iml_ssl:/netscratch/iml_ssl,/ds:/ds:ro,"`pwd`":"`pwd`" \
					--export="NCCL_IB_DISABLE=1" \
					--export="OMP_NUM_THREADS=10" \
					--export="LOGLEVEL=INFO" \
					--export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
					--export="FI_PROVIDER='efa'" \
					--export="CUDA_LAUNCH_BLOCKING=1" bash bashscript/download_data_1.sh 
.PHONY: download_data_1

download_data_2:
	srun -p A100-IML -t 6-23:59:59 --ntasks 1 \
					--mem-per-cpu 40G\
					--container-image=/netscratch/duynguyen/Research/medllm_new/Docker/medical-llama2-tannp-A100.sqsh \
					--container-workdir="`pwd`" \
					--container-mounts=/netscratch/duynguyen:/netscratch/duynguyen,/netscratch/iml_ssl:/netscratch/iml_ssl,/ds:/ds:ro,"`pwd`":"`pwd`" \
					--export="NCCL_IB_DISABLE=1" \
					--export="OMP_NUM_THREADS=10" \
					--export="LOGLEVEL=INFO" \
					--export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
					--export="FI_PROVIDER='efa'" \
					--export="CUDA_LAUNCH_BLOCKING=1" bash bashscript/download_data_2.sh 
.PHONY: download_data_2

download_data_3:
	srun -p A100-IML -t 6-23:59:59 --ntasks 1 \
					--mem-per-cpu 40G\
					--container-image=/netscratch/duynguyen/Research/medllm_new/Docker/medical-llama2-tannp-A100.sqsh \
					--container-workdir="`pwd`" \
					--container-mounts=/netscratch/duynguyen:/netscratch/duynguyen,/netscratch/iml_ssl:/netscratch/iml_ssl,/ds:/ds:ro,"`pwd`":"`pwd`" \
					--export="NCCL_IB_DISABLE=1" \
					--export="OMP_NUM_THREADS=10" \
					--export="LOGLEVEL=INFO" \
					--export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
					--export="FI_PROVIDER='efa'" \
					--export="CUDA_LAUNCH_BLOCKING=1" bash bashscript/download_data_3.sh 
.PHONY: download_data_3

download_data_4:
	srun -p A100-IML -t 6-23:59:59 --ntasks 1 \
					--mem-per-cpu 40G\
					--container-image=/netscratch/duynguyen/Research/medllm_new/Docker/medical-llama2-tannp-A100.sqsh \
					--container-workdir="`pwd`" \
					--container-mounts=/netscratch/duynguyen:/netscratch/duynguyen,/netscratch/iml_ssl:/netscratch/iml_ssl,/ds:/ds:ro,"`pwd`":"`pwd`" \
					--export="NCCL_IB_DISABLE=1" \
					--export="OMP_NUM_THREADS=10" \
					--export="LOGLEVEL=INFO" \
					--export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
					--export="FI_PROVIDER='efa'" \
					--export="CUDA_LAUNCH_BLOCKING=1" bash bashscript/download_data_4.sh 
.PHONY: download_data_4




untar_files:
	srun -p A100-IML -t 6-23:59:59 --ntasks 1 \
					--mem-per-cpu 40G\
					--container-image=/netscratch/duynguyen/Research/medllm_new/Docker/medical-llama2-tannp-A100.sqsh \
					--container-workdir="`pwd`" \
					--container-mounts=/netscratch/duynguyen:/netscratch/duynguyen,/netscratch/iml_ssl:/netscratch/iml_ssl,/ds:/ds:ro,"`pwd`":"`pwd`" \
					--export="NCCL_IB_DISABLE=1" \
					--export="OMP_NUM_THREADS=10" \
					--export="LOGLEVEL=INFO" \
					--export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
					--export="FI_PROVIDER='efa'" \
					--export="CUDA_LAUNCH_BLOCKING=1" python3 untar_files.py 
.PHONY: untar_files

# GPT-4 Assisted Instruct Data Generation
generate_vi_instruction_tuning:
	srun -p A100-IML -t 6-23:59:59 --ntasks 1 \
					--gpus-per-task 1 \
					--cpus-per-gpu=2 \
					--mem-per-cpu 40G\
					--container-image=/netscratch/duynguyen/Research/medllm_new/Docker/llava_med_KL.sqsh \
					--container-workdir="`pwd`" \
					--container-mounts=/netscratch/duynguyen:/netscratch/duynguyen,/netscratch/iml_ssl:/netscratch/iml_ssl,/ds:/ds:ro,"`pwd`":"`pwd`" \
					--export="NCCL_IB_DISABLE=1" \
					--export="OMP_NUM_THREADS=10" \
					--export="LOGLEVEL=INFO" \
					--export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
					--export="FI_PROVIDER='efa'" \
					--export="CUDA_LAUNCH_BLOCKING=1" python llava/instruct/instruct_generate.py --input_path data/instruct/llava_med_instruct_fig_captions.json --output_path data/instruct/llava_med_instruct_60k_inline_mentions_gen.jsonl --max_size 60000 --use_inline_mentions True

.PHONY: generate_vi_instruction_tuning

postprocess:
	srun -p A100-IML -t 6-23:59:59 --ntasks 1 \
					--gpus-per-task 1 \
					--cpus-per-gpu=2 \
					--mem-per-cpu 40G\
					--container-image=/netscratch/duynguyen/Research/medllm_new/Docker/llava_med_KL.sqsh \
					--container-workdir="`pwd`" \
					--container-mounts=/netscratch/duynguyen:/netscratch/duynguyen,/netscratch/iml_ssl:/netscratch/iml_ssl,/ds:/ds:ro,"`pwd`":"`pwd`" \
					--export="NCCL_IB_DISABLE=1" \
					--export="OMP_NUM_THREADS=10" \
					--export="LOGLEVEL=INFO" \
					--export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
					--export="FI_PROVIDER='efa'" \
					--export="CUDA_LAUNCH_BLOCKING=1" python llava/instruct/instruct_postprocess.py --input_path data/instruct/llava_med_instruct_60k_inline_mentions_gen.json --output_path data/instruct/llava_med_instruct_60k_inline_mentions_post.json
.PHONY: postprocess

# Training
initialize_llava_med:
	srun -p A100-IML -t 6-23:59:59 --ntasks 1 \
					--gpus-per-task 1 \
					--cpus-per-gpu=2 \
					--mem-per-cpu 40G\
					--container-image=/netscratch/duynguyen/Research/medllm_new/Docker/llava_med_KL.sqsh \
					--container-workdir="`pwd`" \
					--container-mounts=/netscratch/duynguyen:/netscratch/duynguyen,/netscratch/iml_ssl:/netscratch/iml_ssl,/ds:/ds:ro,"`pwd`":"`pwd`" \
					--export="NCCL_IB_DISABLE=1" \
					--export="OMP_NUM_THREADS=10" \
					--export="LOGLEVEL=INFO" \
					--export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
					--export="FI_PROVIDER='efa'" \
					--export="CUDA_LAUNCH_BLOCKING=1" python3 -m llava.model.apply_delta --base-model-path /netscratch/duynguyen/Research/medllm_new/llama2/weights/models--epfl-llm--meditron-7b/snapshots/073d428e97e1a52726c711b8226baf6247846322 --target-model-path output/LLaVA-7b-v0 --delta-path liuhaotian/LLaVA-7b-delta-v0

.PHONY: initialize_llava_med

# Stage 1
stage1:
	srun -p A100-IML -t 6-23:59:59 --ntasks 1 \
					--gpus-per-task 2 \
					--cpus-per-gpu=5 \
					--mem-per-cpu 20G\
					--container-image=/netscratch/duynguyen/Research/medllm_new/Docker/llava_med_KL.sqsh \
					--container-workdir="`pwd`" \
					--container-mounts=/netscratch/duynguyen:/netscratch/duynguyen,/netscratch/iml_ssl:/netscratch/iml_ssl,/ds:/ds:ro,"`pwd`":"`pwd`" \
					--export="NCCL_IB_DISABLE=1" \
					--export="OMP_NUM_THREADS=10" \
					--export="LOGLEVEL=INFO" \
					--export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
					--export="FI_PROVIDER='efa'" \
					--export="CUDA_LAUNCH_BLOCKING=1" --export="CUDA_VISIBLE_DEVICES=0,1" torchrun --nproc_per_node=2 --master_port=25001 llava/train/train_mem.py \
					--model_name_or_path output/LLaVA-7b-v0 \
					--data_path /netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LLaVA-Med/data/alignment/llava_med_alignment_new.json \
					--image_folder /netscratch/duynguyen/Research/Nghiem_LLaVA-Med/LLaVA-Med/data/images \
					--vision_tower openai/clip-vit-large-patch14 \
					--tune_mm_mlp_adapter True \
					--mm_vision_select_layer -2 \
					--mm_use_im_start_end \
					--bf16 True \
					--output_dir /output/llava-med-7b-pretrain \
					--num_train_epochs 1 \
					--per_device_train_batch_size 2 \
					--per_device_eval_batch_size 4 \
					--gradient_accumulation_steps 8 \
					--evaluation_strategy "no" \
					--save_strategy "steps" \
					--save_steps 2400 \
					--save_total_limit 1 \
					--learning_rate 2e-3 \
					--weight_decay 0. \
					--warmup_ratio 0.03 \
					--lr_scheduler_type "cosine" \
					--logging_steps 1 \
					--model_max_length 2048 \
					--gradient_checkpointing True \
					--lazy_preprocess True \
					--report_to wandb
.PHONY: stage1 

# Stage 2
stage2:
	srun -p A100-IML -t 6-23:59:59 --ntasks 1 \
					--gpus-per-task 2 \
					--cpus-per-gpu=2 \
					--mem-per-cpu 40G\
					--container-image /netscratch/duynguyen/Research/medllm_new/Docker/llava_med_KL.sqsh \
					--container-workdir="`pwd`" \
					--container-mounts /netscratch/duynguyen:/netscratch/duynguyen,/netscratch/iml_ssl:/netscratch/iml_ssl,/ds:/ds:ro,"`pwd`":"`pwd`" \
					--export="NCCL_IB_DISABLE=1" \
					--export="OMP_NUM_THREADS=10" \
					--export="LOGLEVEL=INFO" \
					--export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
					--export="FI_PROVIDER='efa'" \
					--export="CUDA_LAUNCH_BLOCKING=1" bash bashscript/stage2.sh
.PHONY: stage2

finetuning_pathvqa:
	srun -p A100-IML -t 6-23:59:59 --ntasks 1 \
					--gpus-per-task 1 \
					--cpus-per-gpu=2 \
					--mem-per-cpu 40G\
					--container-image= /netscratch/trnguyen/container/llava_med_KL.sqsh \
					--container-workdir="`pwd`" \
					--container-mounts=/netscratch/trnguyen:/netscratch/trnguyen,/netscratch/iml_ssl:/netscratch/iml_ssl,/ds:/ds:ro,"`pwd`":"`pwd`" \
					--export="NCCL_IB_DISABLE=1" \
					--export="OMP_NUM_THREADS=10" \
					--export="LOGLEVEL=INFO" \
					--export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
					--export="FI_PROVIDER='efa'" \
					--export="CUDA_LAUNCH_BLOCKING=1" python3 LVLM-Med/llava/train/train_mem.py \
    --model_name_or_path /netscratch/trnguyen/llava_med_checkpoints_llama/llava_med_in_text_60k_ckpt2 \
    --data_path /netscratch/trnguyen/PathVQA1/data/train.json \
    --image_folder /netscratch/trnguyen/PathVQA1/data/images \
    --vision_tower openai/clip-vit-large-patch14 \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end True \
    --bf16 True \
    --output_dir /netscratch/trnguyen/llama_checkpoint/eval/fine_tuned/pathvqa1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 5000 \
    --save_total_limit 3 \
    --learning_rate 2e-5 \
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
    --report_to wandb

.PHONY: finetuning_pathvqa

#Finetuning on VQA-RAD
finetuning_vqarad: 
	srun -p A100-IML --time=1-00:00 \
	--ntasks 1 \
	--gpus-per-task 2 \
	--cpus-per-gpu=8 \
	--mem-per-cpu 8G \
	--container-image=/netscratch/trnguyen/container/llava_med_KL.sqsh \
	--container-workdir="/netscratch/trnguyen/LVLM-Med" \
	--container-mounts=/netscratch/trnguyen:/netscratch/trnguyen,/ds:/ds:ro,/netscratch/software:/netscratch/software:ro \
	--export="NCCL_IB_DISABLE=1" \
	--export="OMP_NUM_THREADS=10" \
	--export="LOGLEVEL=INFO" \
	--export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
	--export="FI_PROVIDER='efa'" \
	--export="CUDA_LAUNCH_BLOCKING=1" bash bashscript/finetuning_vqarad.sh
.PHONY: finetuning_vqarad

#Generate response and evaluate on VQA-RAD
evaluate_vqarad: 
	srun -p A100-IML --time=1-00:00 \
	--ntasks 1 \
	--gpus-per-task 2 \
	--cpus-per-gpu=8 \
	--mem-per-cpu 8G \
	--container-image=/netscratch/trnguyen/container/llava_med_KL.sqsh \
	--container-workdir="/netscratch/trnguyen/LVLM-Med" \
	--container-mounts=/netscratch/trnguyen:/netscratch/trnguyen,/ds:/ds:ro,/netscratch/software:/netscratch/software:ro \
	--export="NCCL_IB_DISABLE=1" \
	--export="OMP_NUM_THREADS=10" \
	--export="LOGLEVEL=INFO" \
	--export="LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH" \
	--export="FI_PROVIDER='efa'" \
	--export="CUDA_LAUNCH_BLOCKING=1" bash bashscript/evaluate_vqarad.sh
.PHONY: evaluate_vqarad