<div align='center'>

## Exgra-Med and LLaVA-Med on CoT Dataset

</div>

### Usage
To run the model settings on our dataset, check running files in folder bashscript. Then, download the Exgra-Med weight from [link](https://exgra-med.github.io/) and put into ``models`` folder before running. Because the Exgra-Med and LLaVA-Med use the same codebase for finetuning, you can change variables ``model_name_or_path`` (for original weight) and ``version`` (for version running) in running ``/sh`` scripts to finetune LLaVA-Med on this dataset. The checkpoint for LLaVA-Med can be also downloaded in this [link](https://exgra-med.github.io/).

In these files in ``bashscript`` folder, we can change settings.
```Shell
export WANDB_API_KEY= #<your_wandb_key>

lr=2e-5
epochs=3
batchsize=16
prompt_mode=simple # cot for running CoT mode or simple for running original mode
use_rag=true # True if use RAG to change to RAG system prompt, else False
version=_exgra_med_on_No_CoT_dataset_epochs${epochs}_batchsize${batchsize}_lr${lr}_prompt_mode_${prompt_mode}_useRAG${use_rag}
model_name_or_path=./models/exgra-med # Path for original Exgra-Med weight
output_dir=./weights_finetuned/CoT-100${version} # Output checkpoint
run_name=CoT-100${version}
answers_file=./test_answer/CoT-100${version}.jsonl # Output predicted answer file

train_file_json=./data/llava_med_mri_bbox_train_CoT.json # Training dataset path
test_file_json=./data/llava_med_mri_bbox_test_CoT.json # Testing dataset path
image_folder=./data/images # Image folder
```
