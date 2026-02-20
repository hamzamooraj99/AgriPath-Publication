import torch
import pytorch_lightning as pl
from torchmetrics import Precision, Recall, F1Score, ConfusionMatrix
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import pandas as pd
import os
import wandb

# --- CONFIGURATION ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_CLASSES = 65
PROJECT_NAME = "AgriPath-VLM-Eval"

class AgriPathDataModule(pl.LightningDataModule):
    def __init__(self, hf_repo, batch_size):
        super().__init__()
        self.hf_repo = hf_repo
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def setup(self, stage=None):
        self.dataset = load_dataset(self.hf_repo)
        self.train_set = self.dataset['train']
        self.val_set = self.dataset['validation']
        self.test_set = self.dataset['test']
        self.lab_test = self.test_set.filter(lambda sample: sample['source']=='lab', num_proc=8).shuffle(seed=42)
        self.field_test = self.test_set.filter(lambda sample: sample['source']=='field', num_proc=8).shuffle(seed=42)
        self.label_idx = {sample['crop_disease_label']: sample['numeric_label'] for sample in self.test_set}
        self.idx_label = {v: k for k, v in self.label_idx.items()}
    
    def collate_fn(self, batch):
        images = [self.transform(sample['image'].convert('RGB')) for sample in batch]
        labels = [sample['numeric_label'] for sample in batch]

        # print(f"Image batch shape: {[image.shape for image in images]}")
        # print(f"Label batch shape: {labels}")

        return torch.stack(images), torch.tensor(labels)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, collate_fn=self.collate_fn, num_workers=10, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=10, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=10, persistent_workers=True)
    
    def lab_loader(self):
        return DataLoader(self.lab_test, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=10, persistent_workers=True)
    
    def field_loader(self):
        return DataLoader(self.field_test, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=10, persistent_workers=True)
    
    def return_labels(self):
        return self.label_idx, self.idx_label

# --- EVLUATION LOGIC ---
def eval_baseline_loop(data_loader, baseline_type, num_classes, majority_class_idx=None):
    #region ====Eval Setup
    # Overall Metrics (Macro Avg)
    pr = Precision(task='multiclass', num_classes=65, average='macro').to(DEVICE)
    re = Recall(task='multiclass', num_classes=65, average='macro').to(DEVICE)
    f1 = F1Score(task='multiclass', num_classes=65, average='macro').to(DEVICE)
    cm = ConfusionMatrix(task='multiclass', num_classes=65).to(DEVICE)

    # Per-Class Metrics
    f1_pClass = F1Score(task='multiclass', num_classes=65, average='none').to(DEVICE)
    pr_pClass = Precision(task='multiclass', num_classes=65, average='none').to(DEVICE)
    re_pClass = Recall(task='multiclass', num_classes=65, average='none').to(DEVICE)
    #endregion

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating on subset", unit="batch"):
            _, y_true = batch
            y_true = y_true.to(DEVICE)

            if baseline_type == "majority":
                y_pred = torch.full_like(y_true, fill_value=majority_class_idx)
            elif baseline_type == 'random':
                y_pred = torch.randint(0, num_classes, y_true.shape, device=DEVICE)
            
            if y_pred.numel() > 0:
                pr.update(y_pred, y_true)
                re.update(y_pred, y_true)
                f1.update(y_pred, y_true)
                cm.update(y_pred, y_true)
                f1_pClass.update(y_pred, y_true)
                pr_pClass.update(y_pred, y_true)
                re_pClass.update(y_pred, y_true)
    
    return {
        'precision': pr.compute().cpu(),
        'recall': re.compute().cpu(),
        'f1_score': f1.compute().cpu(),
        'conf_mat': cm.compute().cpu(),
        'per_class_f1_scores': f1_pClass.compute().cpu(),
        'per_class_pr_scores': pr_pClass.compute().cpu(),
        'per_class_re_scores': re_pClass.compute().cpu(),
        # Add dummy values for parsing metrics to maintain logging consistency
        'false_parse_count': 0,
        'fallback_parse_count': 0,
        'parse_success_rate': 1.0,
        'failed_raw_outputs': []
    }

def evaluate_baseline(baseline_type):
    run_name = f"{baseline_type.title()} Class Baseline"
    print(f"\n{'='*20}\n E V A L U A T I N G: {run_name} \n{'='*20}")

    #region W&B SETUP
    run = wandb.init(
        project=PROJECT_NAME,
        name=run_name,
        config={'model_architecture': baseline_type},
        job_type=f"evaluation_baseline_{baseline_type}"
    )
    #endregion

    print("Preparing Dataset...")
    data_module = AgriPathDataModule("hamzamooraj99/AgriPath-LF16-30k", batch_size=64)
    data_module.setup()
    label_idx, idx_label = data_module.return_labels()

    #region Find Majority Class
    majority_class_idx = 0
    # if baseline_type == 'majority':
    #     train_set = load_dataset("hamzamooraj99/AgriPath-LF16-30k", split='train')
    #     all_labels = torch.tensor([sample['numeric_label'] for sample in train_set])
    #     majority_class_idx = torch.bincount(all_labels).argmax().item()
    #     print(f"Majority class found: Index {majority_class_idx} ({idx_label[majority_class_idx]})")
    #endregion

    #region ==== Data Loaders
    # Using a standard DataLoader, no custom collator needed
    main_loader = data_module.test_dataloader()
    lab_loader = data_module.lab_loader()
    field_loader = data_module.field_loader()
    #endregion

    #region ===== Run Eval
    print("Running evaluation loops...")
    main_metrics = eval_baseline_loop(main_loader, baseline_type, NUM_CLASSES, majority_class_idx)
    lab_metrics = eval_baseline_loop(lab_loader, baseline_type, NUM_CLASSES, majority_class_idx)
    field_metrics = eval_baseline_loop(field_loader, baseline_type, NUM_CLASSES, majority_class_idx)
    #endregion

    #region ==== Log Metrics (Copied and adapted from your VLM script)
    print("Logging results to W&B...")
    def plot_conf_matrix(conf_mat, eval_batch):
        # This helper function is identical
        conf_mat = conf_mat.cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.matshow(conf_mat, cmap=plt.cm.Blues, alpha=0.3)
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                if conf_mat[i, j] > 0:
                    ax.text(x=j, y=i, s=conf_mat[i, j], va='center', ha='center')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f"{run_name}_{eval_batch}")
        return fig
    
    # Log Overall Metrics
    summary_data = [
        ["Main", main_metrics['precision'], main_metrics['recall'], main_metrics['f1_score'], main_metrics['parse_success_rate']],
        ["Lab", lab_metrics['precision'], lab_metrics['recall'], lab_metrics['f1_score'], lab_metrics['parse_success_rate']],
        ["Field", field_metrics['precision'], field_metrics['recall'], field_metrics['f1_score'], field_metrics['parse_success_rate']]
    ]
    summary_columns = ["Source", "Precision", "Recall", "F1 Score", "Parse Success Rate"]
    summary_table = wandb.Table(data=summary_data, columns=summary_columns)
    wandb.log({
        "overall_metrics/Precision": wandb.plot.bar(summary_table, "Source", "Precision", title="Precision Comparison"),
        "overall_metrics/Recall": wandb.plot.bar(summary_table, "Source", "Recall", title="Recall Comparison"),
        "overall_metrics/F1 Score": wandb.plot.bar(summary_table, "Source", "F1 Score", title="F1 Score Comparison"),
        "overall_metrics/Parse Success Rate": wandb.plot.bar(summary_table, "Source", "Parse Success Rate", title="Parse Success Rate Comparison"),
    })

    # Log Confusion Matrix
    main_fig = plot_conf_matrix(main_metrics['conf_mat'], 'main')
    lab_fig = plot_conf_matrix(lab_metrics['conf_mat'], 'lab')
    field_fig = plot_conf_matrix(field_metrics['conf_mat'], 'field')
    wandb.log({
        f"confusion_matrix/{run_name}/main": wandb.Image(main_fig),
        f"confusion_matrix/{run_name}/field": wandb.Image(field_fig),
        f"confusion_matrix/{run_name}/lab": wandb.Image(lab_fig),
    })
    plt.close("all") # Close all figures to free memory

    # Log Per-Class Metrics
    class_names = [idx_label[i] for i in range(NUM_CLASSES)]
    df = pd.DataFrame({
        "Class Name": class_names,
        "F1 (Main)": main_metrics["per_class_f1_scores"].numpy(),
        "F1 (Field)": field_metrics["per_class_f1_scores"].numpy(),
        "F1 (Lab)": lab_metrics["per_class_f1_scores"].numpy(),
    })
    per_class_table = wandb.Table(dataframe=df)
    wandb.log({f"per_class_metrics/{run_name}": per_class_table})

    wandb.finish()
    print(f"--- Finished evaluating {run_name} ---")
    #endregion

if __name__ == "__main__":
    wandb.login(key="9d53025453578d5552c59a417fd34da242216859")

    evaluate_baseline(baseline_type='majority')
    evaluate_baseline(baseline_type='random')
    
    print("\nAll baseline evaluations complete.")