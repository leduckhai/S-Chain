<div align='center'>

## LVLM-Med

</div>

### Usage
To run the model settings on dataset such as VQA-rad, run this command:
```Shell
cd LLaVA-Med
bash llava1-5_stage2_noval_data_rad.sh
```

In this file, we can change settings (to understand more about each setting, Trung has commented on the file.sh):
```Shell
lr=2e-5
version=_newloss_1e0_temp100_afterDe
model_name_or_path=/netscratch/duynguyen/Research/LLaVA-Med/weights_vlap_10/checkpoint_llava_med_instruct_60k_inline_mention_version_1-5${version}
output_dir=/netscratch/duynguyen/Research/LLaVA-Med/weights_finetuned/data-rad-small-10${version}_${lr}
run_name=data_RAD-10${version}_${lr}
answers_file=/netscratch/duynguyen/Research/LLaVA-Med/results_finetuned/data_RAD/ans-opt-small-10${version}_${lr}.jsonl
```

### QAs
#### 1. How can I prepare the downstream datasets ? LLaVA-Med repo does not seem to mention this step. 
Details about the preparation is included in our [DATASET](DATASET.md)
#### 2. I already managed to preprocess the downstream dataset, but the evaluation script run_eval.py requires candidate file. What is it ?
Please see [DATASET](DATASET.md)

#### 3. I used finetuned checkpoint (e.g data_RAD-9epoch_delta.zip) from author to reproduce the number in paper (table 4a), but the numbers are not even close, especially the closed-set accuracy.
The code for accuracy calculation of closed-set QA is incorrect in [run_eval.py](https://github.com/microsoft/LLaVA-Med/blob/main/llava/eval/run_eval.py). Solution: replace the code in [this part](https://github.com/microsoft/LLaVA-Med/blob/main/llava/eval/run_eval.py#L91-L97) by these [code](https://github.com/microsoft/LLaVA-Med/blob/main/llava/eval/run_eval_batch.py#L115-L118) 

#### 4. I finetuned the checkpoint of stage 2 for downstream task following exactly the training configuration on repo LLaVA-Med, but the number does not seem to match number from using finetuned checkpoint of author, and also not match number in the paper. 
Please note that the number provided in the paper LLaVA-Med is by finetuning model for 15 epochs, while the checkpoints provided by the author are only finetuned for 9 epochs (it is shown in the name of checkpoint as well), and the default training configuration on repo is only for 3 epochs. This is the mistake from the author. So one solution is training further, and/or use different batch size. 

