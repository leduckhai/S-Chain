# S-Chain: Structured Visual Chain-of-Thought for Medicine

[![ArXiv](https://img.shields.io/badge/Paper-ArXiv-b31b1b.svg)](https://arxiv.org/pdf/2510.22728)
[![Hugging Face](https://img.shields.io/badge/🤗%20Model-HuggingFace-blue)](https://huggingface.co/datasets/leduckhai/S-Chain)
[![Dataset](https://img.shields.io/badge/📂%20Dataset-S--Chain%20Data-blue)](https://huggingface.co/datasets/leduckhai/S-Chain)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC--BY--NC%204.0-lightgrey.svg)](https://github.com/duyhominhnguyen/S-Chain/blob/main/LICENSE)
[![Website](https://img.shields.io/badge/🌐%20Project%20Page-S--Chain-green)](https://s-chain.github.io/)


## 🔍 What is S-Chain?

S-Chain is the first large-scale dataset of **Structured Visual Chain-of-Thought (SV-CoT)**:
each reasoning step is explicitly linked to visual evidence via bounding boxes.
This enables training and evaluating *grounded* medical VLM reasoning instead of
hallucinated justifications.

- **12,000 medical images** with expert bounding boxes.
- **700k+ VQA / rationale pairs** across **16 languages**.
- Each sample: image, question, answer, stepwise SV-CoT, and per-step visual regions.

We show that supervising VLMs with SV-CoT:
- Improves interpretability
- Improves grounding fidelity (reasoning actually points to the right region)
- Improves robustness across models and languages

<p align="center">
<img src="https://github.com/leduckhai/S-Chain/blob/main/assets/main_pipeline.png" alt="Alt text" width="1400"/>
</p>


## 📣 News

- **[Oct 2025]** Updated experiment scripts and checkpoints for ExGra-Med and LLaVA-Med.
- **[Oct 2025]** Dataset and project site released.

## 🗂 Repo layout

- `s_chain/` — dataset loaders, training loops, evaluation scripts.
- `architectures/` — adapters for each backbone (ExGra-Med, LLaVA-Med, Qwen-VL, InternVL, MedGemma, ...).
- `experiments/` — scripts + configs for all tables/ablations in the paper.
- `medrag_integration/` — Retrieval-Augmented Generation (RAG) setup for medical evidence.
- `data/` — dataset download scripts and directory conventions.

See `INSTALL.md` to install dependencies and download the dataset.

## I. Quickstart

### 1. Install
```bash
git clone https://github.com/leduckhai/S-Chain.git
cd s-chain
bash medrag_integration/install_medrag.sh
pip install -r requirements.txt
pip install -e .
```

### 2. Download the S-Chain dataset

```
cd data
bash download_english.sh        # English-only SV-CoT split
bash download_multilingual.sh   # All 16 languages

```

This will populate:
```
data/
  s_chain_en/
    train.jsonl
    val.jsonl
    test.jsonl
    images/
    annotations/
  s_chain_multilingual/
    ...
```

Each ```*.jsonl``` record contains:

```
{
  "image_path": "images/img_000123.png",
  "question": "...",
  "answer": "...",
  "sv_cot": [
    {
      "step_text": "First, identify the left costophrenic angle...",
      "evidence_bbox": [x, y, w, h]
    },
    {
      "step_text": "Blunting indicates pleural effusion...",
      "evidence_bbox": [x, y, w, h]
    }
  ],
  "language": "en"
}
```

### 📦 Model Checkpoints

| Model                                  | Description                                |🤗 Download Link |
|----------------------------------------|--------------------------------------------|---------------|
| `llava-med-base`                            | LLaVa-Med trained with base settings (Q4 only)                 | [Link](https://huggingface.co/MERGE-Group/llava-med-10)     |
| `llava-med-gpt-cot`                            | LLaVa-Med trained with GPT-synthetic visual COT                   | [Link](https://huggingface.co/MERGE-Group/llava-med-40)     |
| `llava-med-gpt-schain`                            | LLaVa-Med trained with our S-Chain dataset                  | [Link](https://huggingface.co/MERGE-Group/llava-med-40)     |
| `llava-med-gpt-medrag-only`                            | LLaVa-Med with medical retrieval augmented generation and Q4 only                  | [Link](https://huggingface.co/MERGE-Group/llava-med-40)     |
| `llava-med-gpt-medrag-schain`                            | LLaVa-Med with medical retrieval augmented generation and S-Chian                  | [Link](https://huggingface.co/MERGE-Group/llava-med-40)     |
| `exgra-med-base`                            | ExGra-Med trained with base settings (Q4 only)                    | [Link](https://huggingface.co/MERGE-Group/llava-med-10)     |
| `exgra-med-gpt-cot`                            | ExGra-Med trained with GPT-synthetic visual COT                    | [Link](https://huggingface.co/MERGE-Group/llava-med-40)     |
| `exgra-med-gpt-schain`                            |ExGra-Med trained with our S-Chain dataset                     | [Link](https://huggingface.co/MERGE-Group/llava-med-40)     |
| `exgra-med-gpt-medrag-only`                            | ExGra-Med with medical retrieval augmented generation and Q4 only                 | [Link](https://huggingface.co/MERGE-Group/llava-med-40)     |
| `exgra-med-gpt-medrag-schain`                            | ExGra-Med with medical retrieval augmented generation and S-Chian                    | [Link](https://huggingface.co/MERGE-Group/llava-med-40)     |
| `exgra-med-dci-pathvqa`               | Fine-tuned on PATH-VQA                     | [Link](https://huggingface.co/MERGE-Group/exgra-med-dci-pathvqa)     |

<!-- --- -->
Before starting the finetuning/inference/evaluation, download our finetuned checkpoints and put it inside ```architectures/model_name/checkpoints```
<details>
  <summary>Download Checkpoints</summary>

```bash
cd pretrained/
# pip install -U huggingface_hub
# Download MERGE-Group/llava-med-10
huggingface-cli download --resume-download --local-dir-use-symlinks False MERGE-Group/llava-med-10 --local-dir llava-med-10

# Download MERGE-Group/llava-med-40
huggingface-cli download --resume-download --local-dir-use-symlinks False MERGE-Group/llava-med-40 --local-dir llava-med-40

# Download MERGE-Group/exgra-med-10
huggingface-cli download --resume-download --local-dir-use-symlinks False MERGE-Group/exgra-med-10 --local-dir exgra-med-10

# Download MERGE-Group/exgra-med-40
huggingface-cli download --resume-download --local-dir-use-symlinks False MERGE-Group/exgra-med-40 --local-dir exgra-med-40

```

</details>


## II. Run inference with a pretrained checkpoint
Below: load **ExGra-Med** fine-tuned on SV-CoT from Hugging Face and generate answer and grounded rationale.

```
python experiments/run_infer_demo.py \
    --config s_chain/configs/exgra_med/sft_sv_cot.yaml \
    --image_path path/to/chest_xray.png \
    --question "Is there evidence of pneumothorax?"
```
`
**Outputs** will include (a) predicted answer, (b) stepwise visual chain-of-thought, and (c)  bounding boxes per step (saved overlay in ```outputs/viz/```).

## III. 🧪 Reproducing experiments
We evaluate the following training regimes for each backbone:

- **Baseline CoT**: Supervise on model with input image, question and final prediction (Q4).

- **GPT-Synthetic CoT**: Supervise on GPT-based synthetic visual chain-of-thought.

- **SV-CoT (Ours)**: Supervise on our Structured Visual CoT, where each step links to image regions.

- **Medical RAG-only** Fine-tune with medical Retrieval-Augmented Generation context. We follow the techniques by [MIRIAD](https://med-miriad.github.io/) to generate addtional context in input promots and train the models
without our SV-CoT supervision.

- **SV-CoT + RAG (Joint)**: Fine-tune using both: visual step grounding from S-Chain and retrieved evidence from MIRIAD.

All training/eval configs for each model live in ```s_chain/configs/<model_name>/```.

To **train** a model (e.g., **LLAVA-Med**) with any setting:

```
python experiments/run_finetune.py \
    --config s_chain/configs/llava_med/rag_plus_sv_cot.yaml \
    --output_dir runs/llava_med/rag_plus_sv_cot/
```

To **evaluate**:

```
python experiments/run_eval.py \
    --checkpoint runs/llava_med/rag_plus_sv_cot/ckpt_final.pt \
    --split test
```


## Citation
If you find this work useful, please cite our paper:

```
@article{leduc2025schain,
  title={S-Chain: Structured Visual Chain-of-Thought for Medicine},
  author={Le-Duc, Khai and Trinh, Phuong T. H. and Nguyen, Duy M. H. and Nguyen, Tien-Phat and Diep, Nghiem T. and Ngo, An and Vu, Tung and Vuong, Trinh and Nguyen, Anh-Tien and Nguyen, Mau and Hoang, Van Trung and Nguyen, Khai-Nguyen and Nguyen, Hy and Ngo, Chris and Liu, Anji and Ho, Nhat and Hauschild, Anne-Christin and Nguyen, Khanh Xuan and Nguyen-Tang, Thanh and Xie, Pengtao and Sonntag, Daniel and Zou, James and Niepert, Mathias and Nguyen, Anh Totti},
  journal={arXiv preprint},
  year={2025}
}

```

## ⚖️ Important Notice on Dataset Usage

The S-Chain dataset is provided solely for research and educational purposes.
It may contain human or machine annotation errors, as well as potential biases or inconsistencies inherent to medical data.
Users are expected to exercise appropriate caution in interpretation and ensure ethical and non-commercial use.









