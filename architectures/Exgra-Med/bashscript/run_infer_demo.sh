prompt_mode=cot
use_rag=true
output_dir=./checkpoints/exgra-med-gpt-medrag-schain
answers_file=./test_answer/exgra-med-gpt-medrag-schain.jsonl

test_file_json=./data/llava_med_mri_bbox_test_CoT.json
image_folder=./data/images

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