# <img src="https://github.com/leduckhai/S-Chain/blob/main/assets/SChain_icon.png" alt="Logo" width="60" valign="middle"> S-Chain: Structured Visual Chain-of-Thought for Medicine

[![ArXiv](https://img.shields.io/badge/Paper-ArXiv-b31b1b.svg)](https://arxiv.org/abs/2510.22728)
[![Hugging Face](https://img.shields.io/badge/🤗%20Model-HuggingFace-blue)](https://huggingface.co/leduckhai/S-Chain)
[![Dataset](https://img.shields.io/badge/📂%20Dataset-S--Chain%20Data-blue)](https://huggingface.co/datasets/leduckhai/S-Chain)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/leduckhai/S-Chain/blob/main/DATASET_LICENSE.md)
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

- **[Oct 2025]** Updated experiment scripts and checkpoints for ExGra-Med and LLaVA-Med. See the [readme](architectures/Exgra-Med-CoT/README.md) for detailed instructions.
- **[Oct 2025]** Dataset and project site released.

## 🗂 Repo layout

- `architectures/` — adapters for each backbone (ExGra-Med, LLaVA-Med, InternVL, MedGemma, ...). Each model has its own installation and usage instructions.
- `medrag_integration/` — Retrieval-Augmented Generation (RAG) setup for medical evidence.
- `data/` — dataset download scripts and directory conventions.


## I. Quickstart

### 1. Download the S-Chain dataset

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

### 2. Choose a Model Architecture

Each architecture in `architectures/` has its own README with detailed installation, training, and evaluation instructions:

- **[ExGra-Med & LLaVA-Med](architectures/Exgra-Med-CoT/README.md)** — Currently available
- More architectures coming soon...

### 3. Using Pre-trained Models
- **ExGra-Med and LLava-Med**: ...TBU


## II. 📦 Model Checkpoints

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



## III. 🧪 Reproducing experiments
We evaluate the following training regimes for each backbone:

- **Baseline CoT**: Supervise on model with input image, question and final prediction (Q4).

- **GPT-Synthetic CoT**: Supervise on GPT-based synthetic visual chain-of-thought.

- **SV-CoT (Ours)**: Supervise on our Structured Visual CoT, where each step links to image regions.

- **Medical RAG-only** Fine-tune with medical Retrieval-Augmented Generation context. We follow the techniques by [MIRIAD](https://med-miriad.github.io/) to generate addtional context in input promots and train the models
without our SV-CoT supervision.

- **SV-CoT + RAG (Joint)**: Fine-tune using both: visual step grounding from S-Chain and retrieved evidence from MIRIAD.


### Currently Available Models

**ExGra-Med & LLaVA-Med**
```bash
cd architectures/Exgra-Med-CoT
bash bashscript/llava1-5_stage2_noval_CoT.sh
```
See [architectures/Exgra-Med-CoT/README.md](architectures/Exgra-Med-CoT/README.md) for detailed configuration options.

**More architectures coming soon...**


## Citation
If you find this work useful, please cite our paper: [https://arxiv.org/abs/2510.22728](https://arxiv.org/abs/2510.22728)

```
@article{leduc2025schain,
  title={S-Chain: Structured Visual Chain-of-Thought For Medicine},
  author={Le-Duc, Khai and Trinh, Phuong T. H. and Nguyen, Duy M. H. and Nguyen, Tien-Phat and Diep, Nghiem T. and Ngo, An and Vu, Tung and Vuong, Trinh and Nguyen, Anh-Tien and Nguyen, Mau and Hoang, Van Trung and Nguyen, Khai-Nguyen and Nguyen, Hy and Ngo, Chris and Liu, Anji and Ho, Nhat and Hauschild, Anne-Christin and Nguyen, Khanh Xuan and Nguyen-Tang, Thanh and Xie, Pengtao and Sonntag, Daniel and Zou, James and Niepert, Mathias and Nguyen, Anh Totti},
  journal={arXiv preprint},
  eprint={2510.22728},
  url={https://arxiv.org/abs/2510.22728},
  year={2025}
}
```

## ⚖️ Important Notice on Dataset Usage

The S-Chain dataset is provided solely for research and educational purposes.
It may contain human or machine annotation errors, as well as potential biases or inconsistencies inherent to medical data.
Users are expected to exercise appropriate caution in interpretation and ensure ethical and non-commercial use.









