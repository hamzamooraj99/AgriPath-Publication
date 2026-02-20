#region IMPORTS
import os
import argparse
from typing import Dict, List, Tuple
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from torchmetrics import Precision, Recall, F1Score, ConfusionMatrix, Accuracy
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="google/siglip-base-patch16-224")
parser.add_argument("--model", type=str, required=True, choices=["SigLIP", "CLIP"])
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#endregion

#region TEMPLATES
DISEASED_TEMPLATES = [
    "a photo of a {crop} leaf with {disease}",
    "an image of a {crop} leaf affected by {disease}",
    "a close-up photo of a {crop} leaf showing {disease}",
]

HEALTHY_TEMPLATES = [
    "a photo of a healthy {crop} leaf",
    "an image of a healthy {crop} leaf",
    "a close-up photo of a healthy {crop} leaf",
]
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
test_set = load_dataset("hamzamooraj99/AgriPath-LF16-30k", split='test').shuffle(seed=42)
# Separate dataset via source
field_set = test_set.filter(lambda sample: sample['source']=='field', num_proc=8).shuffle(seed=42)
lab_set = test_set.filter(lambda sample: sample['source']=='lab', num_proc=8).shuffle(seed=42)

class_labels = sorted(set(test_set["crop_disease_label"]))
label2id = {label: i for i, label in enumerate(class_labels)}
id2label = {i: label for label, i in label2id.items()}
#endregion



#region HELPER: CROP-DISEASE
def get_crop_disease_for_label(dataset, crop_disease_label) -> Tuple[str, str]:
    idx = dataset['crop_disease_label'].index(crop_disease_label)
    crop = dataset[idx]['crop']
    disease = dataset[idx]['disease']
    return crop, disease
#endregion

#region HELPER: PROMPT-ENSEMBLE
def build_prompts_for_label(crop, disease) -> List[str]:
    disease_text = str(disease).replace("_", " ").strip()
    crop_text = str(crop).replace("_", " ").strip()

    if disease_text.lower() == "healthy":
        return [t.format(crop=crop_text) for t in HEALTHY_TEMPLATES]
    else:
        return [t.format(crop=crop_text, disease=disease_text) for t in DISEASED_TEMPLATES]
#endregion

#region PROMPT DICTIONARY
prompts_per_class: Dict[str, List[str]] = {}
for label in class_labels:
    crop, disease = get_crop_disease_for_label(test_set, label)
    prompts_per_class[label] = build_prompts_for_label(crop, disease)
#endregion

#region COLLATOR
def collate_fn(batch):
    images = [x["image"].convert("RGB") for x in batch]
    y = [label2id[x["crop_disease_label"]] for x in batch]
    return images, torch.tensor(y, dtype=torch.long)
#endregion

#region EVAL LOOP
@torch.no_grad()
def eval(data_source, source_name: str, processor, model, prototypes):
    loader = DataLoader(data_source, batch_size=32, shuffle=False, collate_fn=collate_fn)

    pr = Precision(task='multiclass', num_classes=len(class_labels), average='macro').to(DEVICE)
    re = Recall(task='multiclass', num_classes=len(class_labels), average='macro').to(DEVICE)
    f1 = F1Score(task='multiclass', num_classes=len(class_labels), average='macro').to(DEVICE)
    cm = ConfusionMatrix(task='multiclass', num_classes=len(class_labels)).to(DEVICE)
    bal_acc = Accuracy(task='multiclass', num_classes=len(class_labels), average='macro').to(DEVICE)
    pr_pClass = Precision(task='multiclass', num_classes=len(class_labels), average=None).to(DEVICE)
    re_pClass = Recall(task='multiclass', num_classes=len(class_labels), average=None).to(DEVICE)
    f1_pClass = F1Score(task='multiclass', num_classes=len(class_labels), average=None).to(DEVICE) 

    for images, y in tqdm(loader, desc=f"Evaluating {source_name}", unit="batch"):
        y = y.to(DEVICE)
        inputs = processor(images=images, return_tensors='pt').to(DEVICE)
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        sims = image_features @ prototypes.T
        preds = torch.argmax(sims, dim=-1)

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
        project="AgriPath-VLM-Eval",
        name=f"{args.model}-ZS",
        config={
            "checkpoint": args.checkpoint,
            "batch_size": 32,
            "hf_repo": "hamzamooraj99/AgriPath-LF16-30k",
            "templates_diseased": DISEASED_TEMPLATES,
            "templates_healthy": HEALTHY_TEMPLATES,
            "method": f"{args.model.lower()}_zeroshot_template_ensemble",
        },
        job_type=f"ZS_{args.model.lower()}",
    )

    #region MAIN: Model Loading
    processor = AutoProcessor.from_pretrained(args.checkpoint)
    model = AutoModelForZeroShotImageClassification.from_pretrained(args.checkpoint).to(DEVICE)
    model.eval()
    #endregion

    #region BUILD CLASS PROTO
    @torch.no_grad()
    def build_class_prototypes() -> torch.Tensor:
        prototypes = []
        for label in class_labels:
            texts = prompts_per_class[label]
            inputs = processor(text=texts, return_tensors='pt', padding=True, truncation=True).to(DEVICE)
            text_features = model.get_text_features(**inputs)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            prototype = text_features.mean(dim=0, keepdim=True)
            prototype = prototype / prototype.norm(dim=-1, keepdim=True)
            prototypes.append(prototype)
        
        return torch.cat(prototypes, dim=0)

    CLASS_PROTOTYPES = build_class_prototypes()
    #endregion

    main_metrics = eval(test_set, "main", processor, model, CLASS_PROTOTYPES)
    lab_metrics = eval(lab_set, "lab", processor, model, CLASS_PROTOTYPES)
    field_metrics = eval(field_set, "field", processor, model, CLASS_PROTOTYPES)

    print("LOGGING TO W&B")

    summary_data = [
        ["Main", main_metrics['balanced_accuracy'], main_metrics['precision'], main_metrics['recall'], main_metrics['f1_score']],
        ["Lab", lab_metrics['balanced_accuracy'], lab_metrics['precision'], lab_metrics['recall'], lab_metrics['f1_score']],
        ["Field", field_metrics['balanced_accuracy'], field_metrics['precision'], field_metrics['recall'], field_metrics['f1_score']]
    ]
    summary_columns = ["Source", "Balanced Accuracy", "Precision", "Recall", "F1 Score"]
    summary_table = wandb.Table(data=summary_data, columns=summary_columns)
    wandb.log({
        "overall_metrics/Balanced Accuracy": wandb.plot.bar(summary_table, "Source", "Balanced Accuracy"),
        "overall_metrics/Precision": wandb.plot.bar(summary_table, "Source", "Precision"),
        "overall_metrics/Recall": wandb.plot.bar(summary_table, "Source", "Recall"),
        "overall_metrics/F1 Score": wandb.plot.bar(summary_table, "Source", "F1 Score"),
    })

    # Log Confusion Matrix
    main_fig = plot_conf_matrix(main_metrics['conf_mat'], f"{args.model}", 'main')
    lab_fig = plot_conf_matrix(lab_metrics['conf_mat'], f"{args.model}", 'lab')
    field_fig = plot_conf_matrix(field_metrics['conf_mat'], f"{args.model}", 'field')
    wandb.log({
        f"confusion_matrix/{args.model}/main": wandb.Image(main_fig),
        f"confusion_matrix/{args.model}/field": wandb.Image(field_fig),
        f"confusion_matrix/{args.model}/lab": wandb.Image(lab_fig),
    })
    plt.close(main_fig); plt.close(lab_fig); plt.close(field_fig)

    # Log Per-Class Metrics
    class_names = [id2label[i] for i in range(65)]

    combined_per_class_data = {
        "Class Name": class_names,
        "F1 (Main)": main_metrics["per_class_f1_scores"].numpy(),
        "F1 (Field)": field_metrics["per_class_f1_scores"].numpy(),
        "F1 (Lab)": lab_metrics["per_class_f1_scores"].numpy(),
        "Precision (Main)": main_metrics["per_class_pr_scores"].numpy(),
        "Precision (Field)": field_metrics["per_class_pr_scores"].numpy(),
        "Precision (Lab)": lab_metrics["per_class_pr_scores"].numpy(),
        "Recall (Main)": main_metrics["per_class_re_scores"].numpy(),
        "Recall (Field)": field_metrics["per_class_re_scores"].numpy(),
        "Recall (Lab)": lab_metrics["per_class_re_scores"].numpy(),
    }
    df = pd.DataFrame(combined_per_class_data)
    per_class_table = wandb.Table(dataframe=df)
    wandb.log({
        f"per_class_metrics": per_class_table,
    })

    wandb.finish()

    
if __name__ == "__main__":
    main()