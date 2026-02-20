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
└── README.md
```

---

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
