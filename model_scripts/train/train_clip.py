import os
import json
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

def seed_everything(seed):
    pl.seed_everything(seed, workers=True)

class AgriPathCLIPDataModule(pl.LightningDataModule):
    def __init__(self, hf_repo, processor_name, batch_size=64, num_workers=8, seed=42):
        super().__init__()
        self.hf_repo = hf_repo
        self.processor_name = processor_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        self.processor = AutoProcessor.from_pretrained(self.processor_name)
    
    def prepare_data(self):
        load_dataset(self.hf_repo)
    
    def setup(self, stage=None):
        ds = load_dataset(self.hf_repo)
        self.train_set = ds["train"].shuffle(seed=self.seed)
        self.val_set = ds["validation"].shuffle(seed=self.seed)

    def collate_fn(self, batch):
        images = [x["image"].convert("RGB") for x in batch]
        proc = self.processor(images=images, return_tensors='pt')
        labels = torch.tensor([x["numeric_label"] for x in batch], dtype=torch.long)
        return proc['pixel_values'], labels
    
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, collate_fn=self.collate_fn,)
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, collate_fn=self.collate_fn,)
    
class LinearProbeModel(pl.LightningModule):
    def __init__(self, backbone_name, num_classes=65, lr=1e-3, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        # Infer feature dim from common CLIP/SigLIP configs
        if hasattr(self.backbone, "config"):
            if hasattr(self.backbone.config, "projection_dim") and self.backbone.config.projection_dim is not None:
                feat_dim = int(self.backbone.config.projection_dim)
            elif hasattr(self.backbone.config, "hidden_size") and self.backbone.config.hidden_size is not None:
                feat_dim = int(self.backbone.config.hidden_size)
            elif hasattr(self.backbone.config, "vision_config") and hasattr(self.backbone.config.vision_config, "hidden_size"):
                feat_dim = int(self.backbone.config.vision_config.hidden_size)
            else:
                raise ValueError("Could not infer feature dim from backbone config.")
        else:
            raise ValueError("Backbone has no config; cannot infer feature dim.")

        self.classifier = nn.Linear(feat_dim, num_classes)

        for p in self.backbone.parameters():
            p.requires_grad = False
        self.backbone.eval()

        self.uses_get_image_features = hasattr(self.backbone, "get_image_features")
        if not self.uses_get_image_features and not hasattr(self.backbone, "vision_model"):
            raise ValueError(
                f"Backbone {backbone_name} doesn't expose get_image_features or vision_model; "
                "please use a CLIP/SigLIP-like checkpoint."
            )
        
        # self.classifier = None
        self.criterion = nn.CrossEntropyLoss()
    
    def _image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.uses_get_image_features:
                features = self.backbone.get_image_features(pixel_values=pixel_values)
            else:
                out = self.backbone.vision_model(pixel_values=pixel_values)
                if hasattr(out, "pooler_output") and out.pooler_output is not None:
                    features = out.pooler_output
                else:
                    features = out.last_hidden_state[:, 0, :]
        
        features = features / features.norm(dim=-1, keepdim=True,).clamp(min=1e-12)
        return features
    
    # def _maybe_init_classifier(self, features: torch.Tensor):
    #     if self.classifier is None:
    #         d = features.shape[-1]
    #         self.classifier = nn.Linear(d, self.hparams.num_classes).to(self.device)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        features = self._image_features(pixel_values=pixel_values)
        # self._maybe_init_classifier(features)
        logits = self.classifier(features)
        return logits
    
    def training_step(self, batch, batch_idx):
        pixel_values, y = batch
        pixel_values = pixel_values.to(self.device)
        y = y.to(self.device)

        logits = self(pixel_values)
        loss = self.criterion(logits, y)

        # preds = torch.argmax(logits, dim=-1)
        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        # self.log("train/f1", self.f1(preds, y), prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        pixel_values, y = batch
        pixel_values = pixel_values.to(self.device)
        y = y.to(self.device)

        logits = self(pixel_values)
        loss = self.criterion(logits, y)

        # preds = torch.argmax(logits, dim=-1)
        self.log("val/loss", loss, prog_bar=True, on_epoch=True)
        # self.log("val/f1", self.f1(preds, y), prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW([p for p in self.parameters() if p.requires_grad], lr=self.hparams.lr, weight_decay=self.hparams.weight_decay,)
    
    def export_head(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        head_path = os.path.join(out_dir,"classifier_head.pt")
        meta_path = os.path.join(out_dir, "metadata.json")

        torch.save(self.classifier.state_dict(), head_path)

        metadata = {
            "backbone_name": self.hparams.backbone_name,
            "num_classes": int(self.hparams.num_classes),
            "feature_dim": int(self.classifier.in_features),
            "head_type": "linear",
            "normalization": "l2_normalized_image_features",
            "note": "Backbone frozen; head trained via linear probing"
        }
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        return head_path, meta_path
    
def main(lr):
    """
    --model google/siglip-base-patch16-224 --run_name SigLIP_google_patch16 --base SigLIP
    --model openai/clip-vit-base-patch32 --run_name CLIP_openai_patch32
    --model openai/clip-vit-large-patch14 --run_name CLIP_openai_large14
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--hf_repo", type=str, default="hamzamooraj99/AgriPath-LF16-30k")
    parser.add_argument("--project", type=str, default="AgriPath-VLM")
    parser.add_argument("--run_name", type=str, default=None, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    # parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--out_dir", type=str, default="linear_probe_heads")
    parser.add_argument("--base", type=str, default="CLIP")
    args = parser.parse_args()

    seed_everything(args.seed)

    wandb_logger = WandbLogger(project=args.project, name=f"{args.run_name}_LR{lr}", log_model=False, job_type=f"{args.run_name}_linear_probing")

    wandb_logger.experiment.config.update({
        "learning_rate": lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "backbone": args.model,
        "method": "linear_probe"
    })


    datamodule = AgriPathCLIPDataModule(hf_repo=args.hf_repo, processor_name=args.model, batch_size=args.batch_size, num_workers=args.num_workers, seed=args.seed)
    probe = LinearProbeModel(backbone_name=args.model, num_classes=65, lr=lr, weight_decay=args.weight_decay)

    ckpt = ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, filename="{epoch}-{val_loss:.4f}")

    trainer = pl.Trainer(max_epochs=args.epochs, accelerator="gpu" if torch.cuda.is_available() else "cpu", devices=1, logger=wandb_logger, callbacks=[ckpt], log_every_n_steps=10)
    trainer.fit(probe, datamodule=datamodule)

    if ckpt.best_model_path:
        probe = LinearProbeModel.load_from_checkpoint(ckpt.best_model_path)
        probe = probe.to(trainer.strategy.root_device)
    probe.eval()

    # datamodule.setup()
    # pixel_values, _ = next(iter(datamodule.val_dataloader()))
    # pixel_values = pixel_values.to(probe.device)
    # probe = probe.to(pixel_values.device)
    # _ = probe(pixel_values)

    run = wandb_logger.experiment
    head_path, meta_path = probe.export_head(args.out_dir)

    artifact_name = f"{args.run_name}_LR{lr}"
    artifact = wandb.Artifact(name=artifact_name, type="linear_probe_head")
    artifact.add_file(head_path)
    artifact.add_file(meta_path)
    run.log_artifact(artifact)

    wandb.finish()

if __name__ == "__main__":
    lrs = [1e-3, 3e-3, 1e-2]
    for lr in lrs:
        print(f"Linear Probing with LR={lr}")
        main(lr)
        print("COMPLETE... Moving to next... \n")
