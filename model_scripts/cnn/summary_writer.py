'''
# summary_writer.py
## Author: @hamzamooraj99 (Hamza Hassan Mooraj)
Description: This file contains a script to write a summary of all CNN model experiments using TensorBoard
'''

import torch
# from pytorch_lightning.loggers import TensorBoardLogger
# import pytorch_lightning as pl
import resnet50_lightning as rn
from torchmetrics import Accuracy, Precision, Recall, F1Score, ConfusionMatrix
import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import os
import wandb
from pathlib import Path
import argparse

#region CONFIGURATION
parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--field", action="store_true")
group.add_argument("--lab", action="store_true")
# group.add_argument("--combined", action="store_true")

args = parser.parse_args()

# SCRIPT_DIR = Path(__file__).parent
HF_REPO = "hamzamooraj99/AgriPath-LF16-30k"
NUM_CLASSES = 65
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

DataModule = rn.AgriPathDataModule
ModelModule = rn.ResNet50TLModel

if args.field:
    artifact_path = "hhm2000-heriot-watt-university/AgriPath-VLM/field_cnn_paths:v0"
    job_type = "NE_field_cnn"
elif args.lab:
    artifact_path = "hhm2000-heriot-watt-university/AgriPath-VLM/lab_cnn_paths:v0"
    job_type = "NE_lab_cnn"
# elif args.combined:
#     artifact_path = None
else:
    artifact_path = None
    raise TypeError("Undefined argument. Select from --field, --lab, or --combined")

if not artifact_path:
    raise ValueError("Unexpected value - No Artifact found")
#endregion

#region HELPER FUNCTIONS
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

def download_artifact(artifact_path: str):
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    resolver_run = wandb.init(
        project="AgriPath-VLM",
        job_type="download_artifact"
    )

    artifact = resolver_run.use_artifact(artifact_path, type="model")
    artifact_dir = Path(artifact.download())

    experiments = {
        "ResNet50 Batch=16 LR=1e-4": (artifact_dir / "resnet50_agripath_exp_0.pth", 16, 1e-4),
        "ResNet50 Batch=16 LR=5e-4": (artifact_dir / "resnet50_agripath_exp_1.pth", 16, 5e-4),
        "ResNet50 Batch=16 LR=2e-4": (artifact_dir / "resnet50_agripath_exp_2.pth", 16, 2e-4),
        "ResNet50 Batch=32 LR=1e-4": (artifact_dir / "resnet50_agripath_exp_3.pth", 32, 1e-4),
        "ResNet50 Batch=32 LR=5e-4": (artifact_dir / "resnet50_agripath_exp_4.pth", 32, 5e-4),
        "ResNet50 Batch=32 LR=2e-4": (artifact_dir / "resnet50_agripath_exp_5.pth", 32, 2e-4),
        "ResNet50 Batch=64 LR=1e-4": (artifact_dir / "resnet50_agripath_exp_6.pth", 64, 1e-4),
        "ResNet50 Batch=64 LR=5e-4": (artifact_dir / "resnet50_agripath_exp_7.pth", 64, 5e-4),
        "ResNet50 Batch=64 LR=2e-4": (artifact_dir / "resnet50_agripath_exp_8.pth", 64, 2e-4)
    }
    resolver_run.finish()
    return experiments
#endregion

#region EVALUATION
def evaluate_model(exp_name, path, batch_size, learning_rate, num_classes = NUM_CLASSES):
    print(f"\nEvaluating {exp_name}...")

    run = wandb.init(
        project="AgriPath-VLM-Eval",
        name=exp_name,
        config={'batch_size': batch_size, 'learning_rate': learning_rate, 'model_architecture': "ResNet50"},
        job_type=job_type
    )

    model = ModelModule(num_classes=num_classes, learning_rate=learning_rate)
    checkpoint = torch.load(path, map_location=torch.device(DEVICE), weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    datamodule = DataModule(HF_REPO, batch_size=batch_size)
    datamodule.setup('test')
    test_loader = datamodule.test_dataloader()
    lab_test = datamodule.lab_loader()
    field_test = datamodule.field_loader()
    label_idx, idx_label = datamodule.return_labels()

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

        batch_iter = 0
        with torch.inference_mode():
            for batch in data_loader:
                # print(f"Batch iter {batch_iter}")
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

                batch_iter+=1
            
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
        path, batch, lr = details[0], details[1], details[2]
        evaluate_model(exp_name=exp, path=path, batch_size=batch, learning_rate=lr)
#endregion
