<div align='center'>

## Exgra-Med and LLaVA-Med on CoT Dataset

</div>

### Usage
To run the model settings on our dataset, check all paths and run this file:
```Shell
cd LLaVA-Med
bash bashscript/llava1-5_stage2_noval_CoT.sh
```

In this file, we can change settings.
```Shell
export WANDB_API_KEY=wandb_key

lr=2e-5
epochs=3
batchsize=16
prompt_mode=cot # cot for running CoT mode or simple for running original mode
version=_exgra_med_on_CoT_dataset_epochs${epochs}_batchsize${batchsize}_lr${lr}_prompt_mode_${prompt_mode}
model_name_or_path=./models/exgra_med_weight
output_dir=./weights_finetuned/CoT-100${version}
run_name=CoT-100${version}
answers_file=./test_answer/CoT-100${version}.jsonl
```
