#region IMPORTS
import os
import json
import argparse
from functools import partial

import torch
import torch.nn as nn
import pandas as pd
import matplotlib.pyplot as plt
import wandb

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel

from torchmetrics import Precision, Recall, F1Score, ConfusionMatrix, Accuracy
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#endregion

#region ARGPARSE
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,)
parser.add_argument("--head_artifact", type=str, required=True)
parser.add_argument("--lr", type=str, required=True)
parser.add_argument("--job_type", type=str, required=True)
parser.add_argument("--project", type=str, default="AgriPath-Evals")
parser.add_argument("-d", "--dataset", type=str, default="hamzamooraj99/AgriPath-LF16-30k-CLEAN")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
#endregion

#region CONF_MAT FUNC
def plot_conf_matrix(conf_mat, run_name, eval_batch):
    conf_mat = conf_mat.cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(conf_mat, cmap=plt.cm.Blues)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(x=j, y=i, s=conf_mat[i, j], va='center', ha='center')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f"{run_name}_{eval_batch}")
    return fig
#endregion

#region DATASET HANDLING
test_set = load_dataset(args.dataset, split='test').shuffle(seed=args.seed)
field_set = test_set.filter(lambda sample: sample['source']=='field', num_proc=args.num_workers).shuffle(seed=args.seed)
lab_set = test_set.filter(lambda sample: sample['source']=='lab', num_proc=args.num_workers).shuffle(seed=args.seed)

class_labels = sorted(set(test_set["crop_disease_label"]))
num_classes = len(set(test_set["numeric_label"]))
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}
#endregion

#region LOAD HEAD
def load_head_from_dir(head_dir: str):
    head_path = os.path.join(head_dir, "classifier_head.pt")
    meta_path = os.path.join(head_dir, "metadata.json")

    if not os.path.exists(head_path): raise FileNotFoundError(f"Missing: {head_path}")
    if not os.path.exists(meta_path): raise FileNotFoundError(f"Missing: {meta_path}")

    state = torch.load(head_path, map_location="cpu")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    return state, meta

def load_head_from_wandb(artifact_repo: str, wandb_run):
    art_dir = wandb_run.use_artifact(artifact_repo, type="linear_probe_head").download()
    return load_head_from_dir(art_dir)
#endregion

#region MODEL LOADING
def load_model(wandb_run):
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    backbone = AutoModel.from_pretrained(args.checkpoint).to(DEVICE)
    backbone.eval()
    for p in backbone.parameters():
        p.requires_grad = False
    
    if not hasattr(backbone, "get_image_features") and not hasattr(backbone, "vision_model"): raise ValueError(f"{args.checkpoint} doesn't expose get_image_features or vision_model; use a CLIP/SigLIP checkpoint")

    head_state, head_meta = load_head_from_wandb(args.head_artifact, wandb_run)

    feat_dim = int(head_meta["feature_dim"])
    head_num_classes = int(head_meta["num_classes"])
    if head_num_classes != num_classes: raise ValueError(f"Head expects num_classes={head_num_classes}, but dataset has {num_classes}")

    classifier = nn.Linear(feat_dim, num_classes).to(DEVICE)
    classifier.load_state_dict(head_state)
    classifier.eval()

    return processor, backbone, classifier, feat_dim, head_meta
#endregion

#region DATALOADER
def collate_fn(batch, processor):
    images = [x["image"].convert("RGB") for x in batch]
    proc = processor(images=images, return_tensors="pt")
    y = torch.tensor([int(x["numeric_label"]) for x in batch], dtype=torch.long)
    return proc["pixel_values"], y

def make_loader(dataset, processor):
    return DataLoader(dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=True, pin_memory=True, collate_fn=partial(collate_fn, processor=processor))
#endregion

#region IMAGE FEATURE EXTACT
@torch.no_grad()
def get_image_features(pixel_values: torch.Tensor, backbone) -> torch.Tensor:
    if hasattr(backbone, "get_image_features"):
        features = backbone.get_image_features(pixel_values=pixel_values)
    else:
        out = backbone.vision_model(pixel_values=pixel_values)
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            features = out.pooler_output
        else:
            features = out.last_hidden_state[:, 0, :]
    
    features = features / features.norm(dim=1, keepdim=True).clamp(min=1e-12)
    return features
#endregion

#region EVAL LOOP
@torch.no_grad()
def eval(dataset, source_name, processor, backbone, classifier):
    dataloader = make_loader(dataset=dataset, processor=processor)

    pr = Precision(task='multiclass', num_classes=len(class_labels), average='macro').to(DEVICE)
    re = Recall(task='multiclass', num_classes=len(class_labels), average='macro').to(DEVICE)
    f1 = F1Score(task='multiclass', num_classes=len(class_labels), average='macro').to(DEVICE)
    cm = ConfusionMatrix(task='multiclass', num_classes=len(class_labels)).to(DEVICE)
    bal_acc = Accuracy(task='multiclass', num_classes=len(class_labels), average='macro').to(DEVICE)
    pr_pClass = Precision(task='multiclass', num_classes=len(class_labels), average=None).to(DEVICE)
    re_pClass = Recall(task='multiclass', num_classes=len(class_labels), average=None).to(DEVICE)
    f1_pClass = F1Score(task='multiclass', num_classes=len(class_labels), average=None).to(DEVICE) 

    for pixel_values, y in tqdm(dataloader, desc=f"Evaluating {source_name}", unit="batch"):
        pixel_values = pixel_values.to(DEVICE)
        y = y.to(DEVICE)
        features = get_image_features(pixel_values=pixel_values, backbone=backbone)
        logits = classifier(features)
        preds = torch.argmax(logits, dim=-1)

        pr.update(preds, y)
        re.update(preds, y)
        f1.update(preds, y)
        cm.update(preds, y)
        bal_acc.update(preds, y)
        pr_pClass.update(preds, y)
        re_pClass.update(preds, y)
        f1_pClass.update(preds, y)
    
    precision = pr.compute().cpu()
    recall = re.compute().cpu()
    f1_score = f1.compute().cpu()
    conf_matrix = cm.compute().cpu()
    balanced_accuracy = bal_acc.compute().cpu()
    precision_per_class = pr_pClass.compute().cpu()
    recall_per_class = re_pClass.compute().cpu()
    f1_per_class = f1_pClass.compute().cpu()

    return({
        'balanced_accuracy': balanced_accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'conf_mat': conf_matrix,
        'per_class_f1_scores': f1_per_class,
        'per_class_pr_scores': precision_per_class,
        'per_class_re_scores': recall_per_class,
    })
#endregion

#region MAIN
def main():
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    run = wandb.init(
        project=args.project,
        name=args.lr,
        config={
            "checkpoint": args.checkpoint,
            "hf_repo": args.hf_repo,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "method": "linear_probe_eval",
            "head_source": args.head_artifact,
            "num_classes": num_classes,
            "feature_norm": "l2",
            "job_type": args.job_type,
        },
        job_type=args.job_type,
    )

    processor, backbone, classifier, feat_dim, head_meta = load_model(run)

    wandb.config.update({
        "feature_dim": feat_dim,
        "head_meta": head_meta,
    })

    main_metrics = eval(test_set, "main", processor, backbone, classifier)
    lab_metrics = eval(lab_set, "lab", processor, backbone, classifier)
    field_metrics = eval(field_set, "field", processor, backbone, classifier)

    summary_data = [
        ["Main", main_metrics["balanced_accuracy"], main_metrics["precision"], main_metrics["recall"], main_metrics["f1_score"]],
        ["Lab", lab_metrics["balanced_accuracy"], lab_metrics["precision"], lab_metrics["recall"], lab_metrics["f1_score"]],
        ["Field", field_metrics["balanced_accuracy"], field_metrics["precision"], field_metrics["recall"], field_metrics["f1_score"]],
    ]
    summary_columns = ["Source", "Balanced Accuracy", "Precision", "Recall", "F1 Score"]
    summary_table = wandb.Table(data=summary_data, columns=summary_columns)
    wandb.log({
        "overall_metrics/Balanced Accuracy": wandb.plot.bar(summary_table, "Source", "Balanced Accuracy"),
        "overall_metrics/Precision": wandb.plot.bar(summary_table, "Source", "Precision"),
        "overall_metrics/Recall": wandb.plot.bar(summary_table, "Source", "Recall"),
        "overall_metrics/F1 Score": wandb.plot.bar(summary_table, "Source", "F1 Score"),
    })

    # Confusion matrices
    main_fig = plot_conf_matrix(main_metrics["conf_mat"], args.lr, "main")
    lab_fig = plot_conf_matrix(lab_metrics["conf_mat"], args.lr, "lab")
    field_fig = plot_conf_matrix(field_metrics["conf_mat"], args.lr, "field")
    wandb.log({
        f"confusion_matrix/{args.lr}/main": wandb.Image(main_fig),
        f"confusion_matrix/{args.lr}/lab": wandb.Image(lab_fig),
        f"confusion_matrix/{args.lr}/field": wandb.Image(field_fig),
    })
    plt.close(main_fig); plt.close(lab_fig); plt.close(field_fig)

    # Per-class table
    class_names = [id2label[i] for i in range(num_classes)]
    df = pd.DataFrame({
        "Class Name": class_names,
        "F1 (Main)": main_metrics["per_class_f1_scores"].numpy(),
        "F1 (Lab)": lab_metrics["per_class_f1_scores"].numpy(),
        "F1 (Field)": field_metrics["per_class_f1_scores"].numpy(),
        "Precision (Main)": main_metrics["per_class_pr_scores"].numpy(),
        "Precision (Lab)": lab_metrics["per_class_pr_scores"].numpy(),
        "Precision (Field)": field_metrics["per_class_pr_scores"].numpy(),
        "Recall (Main)": main_metrics["per_class_re_scores"].numpy(),
        "Recall (Lab)": lab_metrics["per_class_re_scores"].numpy(),
        "Recall (Field)": field_metrics["per_class_re_scores"].numpy(),
    })
    wandb.log({"per_class_metrics": wandb.Table(dataframe=df)})

    wandb.finish()
#endregion


if __name__ == "__main__":
    main()