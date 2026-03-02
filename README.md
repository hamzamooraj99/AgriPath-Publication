<<<<<<< HEAD
# A Systematic Study of Visual Model Architectures for Crop Disease Classification

This repository accompanies a research study evaluating multiple architectural paradigms for fine-grained crop disease classification under diverse acquisition conditions.

We provide:

- A domain-aware benchmark dataset (**AgriPath-LF16**)
- A balanced 30k evaluation subset (**AgriPath-LF16-30k**)
- Unified training and evaluation pipelines for:
  - Convolutional Neural Networks (CNNs)
  - Contrastive Vision–Language Models
  - Generative Vision–Language Models

The goal of this repository is full experimental reproducibility under a shared protocol.

---

# 1. Dataset

## AgriPath-LF16

- **111,307 images**
- **16 crops**
- **41 diseases**
- **65 crop–disease pairs**
- Explicit **Lab vs Field** source separation

Available on HuggingFace:

- Full dataset:  
  https://huggingface.co/datasets/hamzamooraj99/AgriPath-LF16  

- Balanced subset (used in experiments):  
  https://huggingface.co/datasets/hamzamooraj99/AgriPath-LF16-30k  

The 30k subset preserves all classes and supports fair evaluation across domain conditions.

---

# 2. Repository Structure
```.
├── analysis/ # Notebooks and scripts used for analysis
│
├── model_scripts/
│ ├── cnn/
│ │ ├── resnet50_lightning.py
│ │ └── summary_writer.py
│ │
│ ├── train/
│ │ ├── configs/
│ │ ├── train_clip.py
│ │ ├── train_peft.py
│ │ └── train_unsloth.py
│ │
│ ├── eval/
│ │ ├── configs/
│ │ ├── baseline_evaluator.py
│ │ ├── eval_clip.py
│ │ ├── eval_peft.py
│ │ ├── eval_vlm.py
│ │ └── zs_eval_clip.py
│
├── LICENSE
=======
# AgriPath: A Systematic Exploration of Architectural Trade-offs for Crop Disease Classification

> **Publication in Progress** — Submitted to *Transactions on Machine Learning Research (TMLR)*

[![Dataset](https://img.shields.io/badge/HuggingFace-Dataset-yellow?logo=huggingface)](https://huggingface.co/datasets/hamzamooraj99/AgriPath-LF16-30k)
[![W&B](https://img.shields.io/badge/Weights%20&%20Biases-Experiments-orange?logo=weightsandbiases)](https://wandb.ai/hhm2000-heriot-watt-university/AgriPath-Paper/overview)
[![License](https://img.shields.io/github/license/hamzamooraj99/AgriPath-Publication)](LICENSE)

---

## Overview

**AgriPath** is a large-scale benchmark study evaluating the full spectrum of modern computer vision architectures for multi-class crop disease classification. We systematically compare three architectural families on a unified 65-class dataset spanning 16 crops and two imaging conditions (controlled laboratory and real-world field):

| Architecture Family | Representative Models |
|---|---|
| Convolutional Neural Networks (CNN) | ResNet-50 (transfer learning) |
| Contrastive Vision-Language Models | CLIP, SigLIP (zero-shot & linear probe) |
| Generative Vision-Language Models (VLM) | SmolVLM-500M, Qwen2.5-VL-3B, Qwen2.5-VL-7B |

Each model family is evaluated under matching conditions, with VLMs further explored across four LoRA fine-tuning regimes to isolate the effect of training data distribution on generalisation.

---

## Dataset

The **AgriPath-LF16-30k** dataset is available on HuggingFace:

```
hamzamooraj99/AgriPath-LF16-30k
```

- **~30,000** images across **65** crop-disease classes
- **16 crops**: Apple, Bell Pepper, Blueberry, Cherry, Corn, Grape, Olive, Orange, Peach, Potato, Raspberry, Rice, Soybean, Squash, Strawberry, Tomato
- **Two sources**: `lab` (controlled, white-background images) and `field` (uncontrolled, real-world images)
- Standard splits: `train` / `validation` / `test`
- Each sample includes: `image`, `crop`, `disease`, `crop_disease_label`, `numeric_label`, `source`

---

## Repository Structure

```
AgriPath-Publication/
├── model_scripts/
│   ├── cnn/                        # ResNet-50 transfer learning
│   │   ├── resnet50_lightning.py   # PyTorch Lightning model & data module
│   │   └── summary_writer.py       # Per-class metric logging
│   ├── train/                      # Training entry points
│   │   ├── train_clip.py           # CLIP/SigLIP linear probe training
│   │   ├── train_peft.py           # SmolVLM LoRA fine-tuning (PEFT/TRL)
│   │   ├── train_unsloth.py        # Qwen2.5-VL LoRA fine-tuning (Unsloth)
│   │   └── configs/                # Per-model YAML configs (full / lab / field / frozen-vision)
│   │       ├── qwen3/
│   │       ├── qwen7/
│   │       └── smol/
│   └── eval/                       # Evaluation entry points
│       ├── baseline_evaluator.py   # Random & majority-class baselines
│       ├── eval_clip.py            # Linear-probe CLIP/SigLIP evaluation
│       ├── zs_eval_clip.py         # Zero-shot CLIP/SigLIP evaluation
│       ├── eval_peft.py            # SmolVLM LoRA evaluation
│       ├── eval_vlm.py             # Qwen2.5-VL LoRA/zero-shot evaluation
│       ├── configs/                # Eval YAML configs (LoRA & zero-shot)
│       │   ├── lora_evals/
│       │   └── zero_shot/
│       └── helper_scripts/
│           ├── count_classes.py
│           └── find_majority.py
├── analysis/
│   ├── clip_inference.ipynb        # CLIP/SigLIP inference & visualisation
│   ├── cnn_inference.ipynb         # ResNet-50 inference & visualisation
│   ├── dataset_figures.ipynb       # Dataset distribution figures
│   ├── resnet50_lightning.py
│   ├── diagrams/                   # Architecture & results diagrams
│   ├── error analysis/
│   │   ├── CLIP-Full(main).csv     # Per-sample error records (CLIP)
│   │   ├── CNN-Full(main).csv      # Per-sample error records (CNN)
│   │   └── conf_mat.ipynb          # Confusion matrix analysis notebook
│   └── parse_outputs/
│       ├── csv_fix.py              # VLM output post-processing
│       ├── qwen3.csv               # Raw Qwen-3B generation outputs
│       ├── qwen7.csv               # Raw Qwen-7B generation outputs
│       └── smol.csv                # Raw SmolVLM generation outputs
>>>>>>> 8580e35 (FULL CLEANUP OF PUBLICATION SCRIPTS)
└── README.md
```

---

<<<<<<< HEAD
# 3. Architectural Paradigms
Three model families are evaluated under a unified protocol.

---

## 3.1 CNN Baseline

- ResNet-50 pretrained on ImageNet
- Transfer learning setup
- Backbone frozen
- Final residual block + classification head trained
- Grid search over:
  - Batch size: {16, 32, 64}
  - Learning rate: {1e-4, 2e-4, 5e-4}

Implemented using PyTorch Lightning.

Training script: `model_scripts/cnn/resnet50_lightning.py`

---

## 3.2 Contrastive Vision-Language Models
- SigLIP (~203M parameters)
- CLIP ViT-L/14 (~427M parameters)

Two settings:

- Zero-shot cosine similarity with template ensemble
- Linear probing on frozen image embeddings

Training script: `model_scripts/train/train_clip.py`

---

## 3.3 Generative Vision–Language Models

- Qwen2.5-VL 3B
- Qwen2.5-VL 7B
- SmolVLM 500M

Training regimes:

- Zero-shot (prompt variants)
- Frozen Vision (LoRA on language only)
- Full LoRA fine-tuning

Backends:

- Qwen models: Unsloth
- SmolVLM: custom PEFT implementation

Training scripts: `train_unsloth.py` & `train_peft.py`

Evaluation scripts: `eval_vlm.py` & `eval_peft.py`

---

# 4. Experimental Protocol

All architectures are evaluated under three regimes:

- Full (Lab + Field)
- Lab-only
- Field-only

Metrics:

- Macro F1 (primary)
- Precision
- Recall
- Parse Success Rate (for generative models)

Generative inference configuration:

- Deterministic decoding
- Temperature = 0
- No beam search
- Max image edge = 512 px
- bf16 precision

---

# 5. Environment

Experiments were conducted on:

- NVIDIA RTX 4090
- NVIDIA A100
- bf16 mixed precision

Install dependencies: `pip install -r requirements.txt`

Core versions:

- torch==2.7.0
- transformers==4.53.1
- unsloth==2025.7.3
- peft==0.16.0
- pytorch-lightning==2.5.0

Full list available in `requirements.txt`.

---

# 6. Reproducing Main Results

## CNN
Training example: `python model_scripts/cnn/resnet50_lightning.py --config model_scripts/train/configs/cnn_full.yaml`  
Evaluation: `python model_scripts/cnn/summary_writer.py`  

---

## Contrastive VLM
Linear probe: `python model_scripts/train/train_clip.py --config model_scripts/train/configs/clip_full.yaml`  
Evaluation: `python model_scripts/eval/eval_clip.py --config model_scripts/eval/configs/clip_full.yaml`  
Zero-shot: `python model_scripts/eval/zs_eval_clip.py --config model_scripts/eval/configs/clip_zs.yaml`  

---

## Generative VLM
Qwen (Unsloth): `python model_scripts/train/train_unsloth.py --config model_scripts/train/configs/qwen3_full.yaml`  
SmolVLM (PEFT): `python model_scripts/train/train_peft.py --config model_scripts/train/configs/smol_full.yaml`  
Evaluation: `python model_scripts/eval/eval_vlm.py --config ...`  

---

# 7. Experiment Tracking

All experiments were logged using Weights & Biases.

Artifacts are being consolidated into a unified project structure.  
Public links will be provided after review.

---

# 8. Reproducibility Scope

This repository enables reproduction of:

- Main paper results
- Appendix ablations
- Zero-shot experiments
- Linear probe experiments
- LoRA sweeps

All configurations are stored under: `model_scripts/train/configs/` & `model_scripts/eval/configs/`

---

# 9. Generative Output Parsing

Generative outputs are programmatically mapped to the 65 canonical crop–disease classes.

- Invalid outputs are assigned to a `false_parse` class
- Empty generations are penalized in F1
- Parse Success Rate (PSR) is reported separately

Regex patterns are defined in the evaluation scripts.

---

# 10. License

Dataset and code are released under the repository license.

---

# 11. Citation

Anonymous during review.
=======
## Models & Training Regimes

### 1. ResNet-50 (CNN Baseline)

Transfer learning from ImageNet pre-trained weights. The convolutional backbone is partially frozen, with a fine-tuned classification head targeting 65 classes. Trained using **PyTorch Lightning**.

```bash
# Example usage
python model_scripts/cnn/resnet50_lightning.py
```

### 2. CLIP / SigLIP (Contrastive VLM)

Two evaluation modes:

- **Zero-shot**: Cosine similarity against text-engineered prompts (e.g. *"a photo of a tomato leaf with early blight"*)
- **Linear probe**: Frozen CLIP/SigLIP vision encoder + trained linear classification head

```bash
# Zero-shot evaluation
python model_scripts/eval/zs_eval_clip.py --checkpoint google/siglip-base-patch16-224 --model SigLIP

# Linear probe training
python model_scripts/train/train_clip.py --config <path/to/config>

# Linear probe evaluation
python model_scripts/eval/eval_clip.py --config <path/to/config>
```

### 3. Vision-Language Models (Generative VLM)

Three VLMs fine-tuned with **LoRA** (rank 128) for structured generation of crop and disease labels:

| Model | Parameters | Training Backend |
|---|---|---|
| SmolVLM-500M-Instruct | 500M | PEFT / TRL |
| Qwen2.5-VL-3B-Instruct | 3B | Unsloth |
| Qwen2.5-VL-7B-Instruct | 7B | Unsloth |

#### Fine-tuning Regimes

Each VLM is trained under four conditions to study the effect of training-data distribution:

| Regime | Training Data | Config Suffix |
|---|---|---|
| `full_lora` | Full dataset (lab + field) | `*_full_lora.yaml` |
| `lab_lora` | Lab images only | `*_lab.yaml` |
| `field_lora` | Field images only | `*_field.yaml` |
| `frozen_vision` | Full dataset, vision encoder frozen | `*_fv.yaml` |

```bash
# VLM training (Unsloth — Qwen models)
python model_scripts/train/train_unsloth.py --config model_scripts/train/configs/qwen3/qwen3_full_lora.yaml

# VLM training (PEFT/TRL — SmolVLM)
python model_scripts/train/train_peft.py --config model_scripts/train/configs/smol/smol_full_lora.yaml

# VLM evaluation
python model_scripts/eval/eval_vlm.py --config model_scripts/eval/configs/lora_evals/qwen3/qwen3_full/...yaml
```

---

## Evaluation Protocol

All models are evaluated on three test splits:

| Split | Description |
|---|---|
| **Full** | Complete held-out test set |
| **Lab** | Test samples from controlled laboratory settings |
| **Field** | Test samples from real-world field conditions |

**Metrics** (all macro-averaged across 65 classes):

- Precision
- Recall
- F1-Score
- Per-class F1
- Confusion Matrix

Experiments are tracked and logged with **Weights & Biases** under the project `AgriPath-Paper`.

---

## Zero-Shot Baselines

Two statistical baselines are included for reference:

- **Random**: Uniform random class prediction
- **Majority**: Predicts the single most frequent class in the training set

```bash
python model_scripts/eval/baseline_evaluator.py
```

---

## Analysis Notebooks

| Notebook | Purpose |
|---|---|
| `analysis/dataset_figures.ipynb` | Dataset class distribution, source breakdown |
| `analysis/cnn_inference.ipynb` | ResNet-50 inference, top-k predictions, error visualisation |
| `analysis/clip_inference.ipynb` | CLIP/SigLIP inference visualisation |
| `analysis/error analysis/conf_mat.ipynb` | Per-model confusion matrix plotting and error analysis |

---

## Installation & Requirements

**Python 3.10+** is recommended. Install core dependencies:

```bash
pip install torch torchvision torchmetrics
pip install transformers datasets peft trl
pip install pytorch-lightning
pip install unsloth
pip install wandb
pip install pandas matplotlib pyyaml
```

> **Note**: Unsloth requires a compatible CUDA environment. See the [Unsloth installation guide](https://github.com/unslothai/unsloth) for GPU setup instructions.

---

## Configuration

All training and evaluation scripts are driven by YAML configuration files. Key fields:

```yaml
model_name: unsloth/Qwen2.5-VL-3B-Instruct   # HuggingFace model identifier
run_name: Qwen-3B_FLoRA                        # W&B run name
trc: false                                     # Test-run check (limit batches)
job_type: full_lora                            # Experiment type tag
r: 128                                         # LoRA rank
learning_rate: 1.5e-4
weight_decay: 0.05
save_repo: hamzamooraj99/AgriPath-Qwen3B-LoRA  # HuggingFace push target (optional)
```

---

## Experiment Tracking

All experiments are logged to **Weights & Biases**. Set your API key before running:

```bash
export WANDB_API_KEY=<your_api_key>
```

Artifacts (model checkpoints, classifier heads) are stored and versioned as W&B artifacts and optionally pushed to HuggingFace Hub.

<!-- ---

## Citation

If you use AgriPath or this codebase in your research, please cite:

```bibtex
@article{agripath2026,
  title   = {A Systematic Exploration of Architectural Trade-offs for Crop Disease Classification},
  author  = {Mooraj, Hamza and others},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  note    = {Under review}
}
``` -->

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE) file in this repository.
>>>>>>> 8580e35 (FULL CLEANUP OF PUBLICATION SCRIPTS)
