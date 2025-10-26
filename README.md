# S-Chain: Structured Visual Chain-of-Thought for Medicine

[![ArXiv](https://img.shields.io/badge/Paper-ArXiv-b31b1b.svg)](https://arxiv.org/pdf/2410.02615v3)
[![Hugging Face](https://img.shields.io/badge/🤗%20Model-HuggingFace-blue)](https://huggingface.co/MERGE-Group)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC--BY--NC%204.0-lightgrey.svg)](https://github.com/duyhominhnguyen/S-Chain/blob/main/LICENSE)
[![Website](https://img.shields.io/badge/🌐%20Project%20Page-S--Chain-green)](https://phatnguyencs.github.io/s-chain/)


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


## II. Run inference with a pretrained checkpoint
Below: load **ExGra-Med** fine-tuned on SV-CoT from Hugging Face and generate answer + grounded rationale.

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









