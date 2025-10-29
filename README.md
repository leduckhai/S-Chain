# <img src="https://github.com/leduckhai/S-Chain/blob/main/assets/SChain_icon.png" alt="Logo" width="60" valign="middle"> S-Chain: Structured Visual Chain-of-Thought for Medicine

[![ArXiv](https://img.shields.io/badge/Paper-ArXiv-b31b1b.svg)](https://arxiv.org/abs/2510.22728)
[![Hugging Face](https://img.shields.io/badge/🤗%20Model-HuggingFace-blue)](https://huggingface.co/leduckhai/S-Chain)
[![Dataset](https://img.shields.io/badge/📂%20Dataset-S--Chain%20Data-blue)](https://huggingface.co/datasets/leduckhai/S-Chain)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/leduckhai/S-Chain/blob/main/DATASET_LICENSE.md)
[![Website](https://img.shields.io/badge/🌐%20Project%20Page-S--Chain-green)](https://s-chain.github.io/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/leduckhai/S-Chain)

---

⭐ **If you find this project helpful, please consider giving it a [star on GitHub](https://github.com/leduckhai/S-Chain)!**  

---

<p align="center">
  <a href="https://github.com/leduckhai" target="_blank"><strong>Khai Le-Duc</strong></a><sup>* 1,2✉</sup>, 
  <a href="https://scholar.google.com/citations?user=_NIyeykAAAAJ&hl=en" target="_blank"><strong>Duy M. H. Nguyen</strong></a><sup>* 3,4,24✉</sup>, 
  <a href="https://scholar.google.com/citations?user=5CbQH_kAAAAJ&hl=en" target="_blank"><strong>Phuong T. H. Trinh</strong></a><sup>* 5</sup>, 
  <strong>Tien-Phat Nguyen</strong><sup>* 6</sup>,  
  Nghiem T. Diep<sup>** 3</sup>, 
  An Ngo<sup>** 7</sup>, 
  Tung Vu<sup>** 8</sup>, 
  <a href="https://scholar.google.com/citations?user=trFdwLkAAAAJ&hl=en" target="_blank"><strong>Trinh Vuong</strong></a><sup>9</sup>, 
  Anh-Tien Nguyen<sup>10,11</sup>,  
  Mau Nguyen<sup>12</sup>, 
  Van Trung Hoang<sup>13</sup>, 
  <a href="https://scholar.google.com/citations?user=IMryD1YAAAAJ&hl=en" target="_blank"><strong>Khai-Nguyen Nguyen</strong></a><sup>14</sup>, 
  <a href="https://scholar.google.com/citations?user=ZAuQIqwAAAAJ&hl=en" target="_blank"><strong>Hy Nguyen</strong></a><sup>15</sup>, 
  Chris Ngo<sup>2</sup>,  
  <a href="https://scholar.google.com/citations?user=k_4zYecAAAAJ&hl=en" target="_blank"><strong>Anji Liu</strong></a><sup>16</sup>, 
  <a href="https://scholar.google.com/citations?user=Xs7cKMwAAAAJ&hl=en" target="_blank"><strong>Nhat Ho</strong></a><sup>17</sup>, 
  <a href="https://scholar.google.com/citations?user=Khifj_MAAAAJ&hl=en" target="_blank"><strong>Anne-Christin Hauschild</strong></a><sup>11</sup>, 
  <a href="https://scholar.google.com/citations?user=SmqouhIAAAAJ&hl=en" target="_blank"><strong>Khanh Xuan Nguyen</strong></a><sup>18</sup>,  
  <a href="https://scholar.google.com/citations?user=UrTlMiwAAAAJ&hl=en" target="_blank"><strong>Thanh Nguyen-Tang</strong></a><sup>19</sup>, 
  <a href="https://scholar.google.com/citations?user=cnncomYAAAAJ&hl=en" target="_blank"><strong>Pengtao Xie</strong></a><sup>20,21</sup>, 
  <a href="https://scholar.google.com/citations?user=v7i6Uz4AAAAJ&hl=en" target="_blank"><strong>Daniel Sonntag</strong></a><sup>3,22</sup>,  
  <a href="https://scholar.google.com/citations?user=23ZXZvEAAAAJ&hl=en" target="_blank"><strong>James Zou</strong></a><sup>23</sup>, 
  <a href="https://scholar.google.com/citations?user=p5vLzq0AAAAJ&hl=en" target="_blank"><strong>Mathias Niepert</strong></a><sup>4,24</sup>, 
  <a href="https://scholar.google.com/citations?user=EQw8d9AAAAAJ&hl=en" target="_blank"><strong>Anh Totti Nguyen</strong></a><sup>25✉</sup>
</p>


<p align="center">
  <em>*Co-first authors; order randomized &nbsp;&nbsp;|&nbsp;&nbsp; **Co-second authors</em><br>
  <em>✉ Corresponding Authors</em>
</p>

<details>
<summary><strong>🎓 Affiliations</strong> (click to expand)</summary>

1. University of Toronto, Canada  
2. Knovel Engineering Lab, Singapore  
3. German Research Centre for Artificial Intelligence  
4. University of Stuttgart, Germany  
5. Chonnam National University, South Korea  
6. Singapore University of Technology and Design  
7. Bucknell University, USA  
8. Concordia University, Canada  
9. Korea University  
10. Justus Liebig University Giessen, Germany  
11. University Medical Center Göttingen, Germany  
12. Japan Advanced Institute of Science and Technology  
13. Hue University, Vietnam  
14. College of William & Mary, USA  
15. Deakin University, Australia  
16. National University of Singapore  
17. University of Texas at Austin, USA  
18. University of California, Berkeley, USA  
19. New Jersey Institute of Technology, USA  
20. University of California San Diego, USA  
21. MBZUAI, UAE  
22. Oldenburg University, Germany  
23. Stanford University, USA  
24. Max Planck Research School for Intelligent Systems (IMPRS-IS), Germany  
25. Auburn University, USA  

</details>

---

<p align="center">
  ✨ In honor of 
  <a href="https://en.wikipedia.org/wiki/H%E1%BA%A3i_Th%C6%B0%E1%BB%A3ng_L%C3%A3n_%C3%94ng" target="_blank"><strong>Hải Thượng Lãn Ông (海上懶翁) – Lê Hữu Trác (黎友晫)</strong></a>, 
  the father of Vietnamese traditional medicine ✨
</p>

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

### 1. 📥 Download the S-Chain dataset

**Example Usage (Python) from Hugging Face**

👉 [https://huggingface.co/datasets/leduckhai/S-Chain](https://huggingface.co/datasets/leduckhai/S-Chain)

```python
from datasets import load_dataset
dataset = load_dataset("leduckhai/S-Chain")
print(dataset)
```

**Or using Bash**

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


### 2. 📦 Choose Model Checkpoints

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


### 3. Run inference with a pretrained checkpoint
Below: load **ExGra-Med** fine-tuned on SV-CoT from Hugging Face and generate answer and grounded rationale.

```
cd architectures/Exgra-Med-CoT

# Then, choosing one of two ways below:
bash bashscript/run_infer_demo.py 
# or
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

python llava/eval/run_eval_CoT.py \
    --gt ${test_file_json} \
    --pred ${answers_file} \
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


### Currently Available Models
To **train** a provided model with any settings, first you need to move into the corresponding folder in ``./architectures`` and follow the README carefully.

**1. ExGra-Med & LLaVA-Med**

To **train**:
```bash
cd architectures/Exgra-Med-CoT
bash bashscript/llava1-5_stage2_noval_CoT.sh
```

To **evaluate**:

```bash
cd architectures/Exgra-Med-CoT

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

python llava/eval/run_eval_CoT.py \
    --gt ${test_file_json} \
    --pred ${answers_file} \
```
Please find more details in [Exgra-Med & LLaVA-Med](architectures/Exgra-Med/README.md).

*More models coming soon...*

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









