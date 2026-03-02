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
└── README.md
```

---

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
