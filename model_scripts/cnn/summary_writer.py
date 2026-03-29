import os
import re
import argparse
from pathlib import Path

import torch
import pandas as pd
import matplotlib.pyplot as plt
import wandb
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
from torchvision import models
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.convnext import ConvNeXt_Tiny_Weights

import cnn_lightning as cnn

#region ARGPARSE
parser = argparse.ArgumentParser()

parser.add_argument("-m", "--model", type=str, choices=['resnet50', 'convnext'], required=True)
parser.add_argument("-e", "--exp", type=str, choices=["main", "lab", "field"], default="main")
parser.add_argument("--org", type=str, required=True)
parser.add_argument("--artifact_version", type=int, default=0)

args = parser.parse_args()
#endregion

#region CONSTANTS
NUM_CLASSES = 65
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ORG = args.org
TRAIN_PROJ = "AgriPath-VLM"
EVAL_PROJ = "AgriPath-Paper"
HF_REPO = "hamzamooraj99/AgriPath-LF16-30k"
#endregion

#region SETUP
artifact_path = f"{ORG}/{TRAIN_PROJ}/{args.model}_{args.exp}_checkpoints:v{args.artifact_version}"
job_type = f"{args.model}_{args.exp}_eval"

DataModule = cnn.AgriPathDataModule
ModelModule = cnn.CNNLightningModel
#endregion

#region HELPERS
def build_backbone(model_name:str):
    if model_name == 'resnet50':
        return models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    elif model_name == 'convnext':
        return models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

def plot_conf_matrix(conf_mat, run_name, eval_split):
    conf_mat = conf_mat.cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.matshow(conf_mat, cmap=plt.cm.Blues)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{run_name}_{eval_split}")
    return fig

def download_artifact(artifact_path: str):
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    
    resolver_run = wandb.init(
        project=TRAIN_PROJ,
        job_type="download_artifact"
    )

    artifact = resolver_run.use_artifact(artifact_path, type="model")
    artifact_dir = Path(artifact.download())

    experiments = {}
    for ckpt_path in sorted(artifact_dir.glob("*.pth")):
        stem_parts = ckpt_path.stem.split("_")
        try:
            lr = float(stem_parts[-2])
            batch_size = int(stem_parts[-1])
        except:
            continue

        model_name = stem_parts[0]
        pretty_model = "ResNet50" if model_name=='resnet50' else "ConvNeXt-Tiny"
        exp_name = f"{pretty_model} Batch={batch_size} LR={lr}"
        experiments[exp_name] = (ckpt_path, batch_size, lr)

    resolver_run.finish()
    return experiments
#endregion

#region EVALUATION
def evaluate_model(exp_name, path, batch_size, learning_rate, model_name, num_classes = NUM_CLASSES):
    print(f"\nEvaluating {exp_name}...")

    run = wandb.init(
        project=EVAL_PROJ,
        name=exp_name,
        config={'batch_size': batch_size, 'learning_rate': learning_rate, 'model_name': model_name},
        job_type=job_type
    )

    backbone = build_backbone(model_name=model_name)
    model = ModelModule(num_classes=num_classes, learning_rate=learning_rate, backbone=backbone)

    checkpoint = torch.load(path, map_location=torch.device(DEVICE), weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    datamodule = DataModule(HF_REPO, batch_size=batch_size)
    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()
    lab_test = datamodule.lab_loader()
    field_test = datamodule.field_loader()
    _, idx_label = datamodule.return_labels()

    #region ====Evaluation Loop
    def eval_loop(data_loader):
        pr = Precision(task='multiclass', num_classes=num_classes, average='macro').to(DEVICE)
        re = Recall(task='multiclass', num_classes=num_classes, average='macro').to(DEVICE)
        f1 = F1Score(task='multiclass', num_classes=num_classes, average='macro').to(DEVICE)
        cm = ConfusionMatrix(task='multiclass', num_classes=num_classes).to(DEVICE)
        bal_acc = Accuracy(task='multiclass', num_classes=num_classes, average='macro').to(DEVICE)
        pr_pClass = Precision(task='multiclass', num_classes=num_classes, average=None).to(DEVICE)
        re_pClass = Recall(task='multiclass', num_classes=num_classes, average=None).to(DEVICE)
        f1_pClass = F1Score(task='multiclass', num_classes=num_classes, average=None).to(DEVICE)

        with torch.inference_mode():
            for batch in data_loader:
                x, y = batch
                x, y = x.to(DEVICE), y.to(DEVICE)
                y_hat = model(x)
                preds = torch.argmax(y_hat, dim=1)

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

    #region ====Logging
    print("Logging...")

    main_metrics = eval_loop(test_loader)
    lab_metrics = eval_loop(lab_test)
    field_metrics = eval_loop(field_test)

    # Log Overall Metrics
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
    main_fig = plot_conf_matrix(main_metrics['conf_mat'], exp_name, 'main')
    lab_fig = plot_conf_matrix(lab_metrics['conf_mat'], exp_name, 'lab')
    field_fig = plot_conf_matrix(field_metrics['conf_mat'], exp_name, 'field')
    wandb.log({
        f"confusion_matrix/{exp_name}/main": wandb.Image(main_fig),
        f"confusion_matrix/{exp_name}/field": wandb.Image(field_fig),
        f"confusion_matrix/{exp_name}/lab": wandb.Image(lab_fig),
    })
    plt.close(main_fig); plt.close(lab_fig); plt.close(field_fig)

    # Log Per-Class Metrics
    class_names = [idx_label[i] for i in range(65)]

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
        f"per_class_metrics/{exp_name}": per_class_table,
    })

    wandb.finish()
    #endregion

#region MAIN
if __name__ == '__main__':
    experiments = download_artifact(artifact_path)
    for exp, details in experiments.items():
        path, batch, lr = details
        evaluate_model(exp_name=exp, path=path, batch_size=batch, learning_rate=lr, model_name=args.model)
#endregion
