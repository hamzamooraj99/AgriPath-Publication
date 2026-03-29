import itertools
from datasets import load_dataset
import torch
import torch.nn as nn
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.convnext import ConvNeXt_Tiny_Weights
import pytorch_lightning as pl
from functools import partial
from argparse import ArgumentParser
import wandb
import os

def seed_everything(seed):
    pl.seed_everything(seed, workers=True)

def agripath_collate(batch, transform):
    images = [transform(sample["image"].convert("RGB")) for sample in batch]
    labels = [sample["numeric_label"] for sample in batch]
    return torch.stack(images), torch.tensor(labels)

# A class to load AgriPath variant, transform according to model specs and create loaders
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

    def prepare_data(self):
        load_dataset(self.hf_repo)

    def setup(self, stage=None):
        self.dataset = load_dataset(self.hf_repo)
        self.train_set = self.dataset['train']
        self.val_set = self.dataset['validation']
        self.test_set = self.dataset['test']
        # Uncomment below for summary_writer.py
        self.lab_test = self.test_set.filter(lambda sample: sample['source']=='lab', num_proc=8).shuffle(seed=42)
        self.field_test = self.test_set.filter(lambda sample: sample['source']=='field', num_proc=8).shuffle(seed=42)
        self.label_idx = {sample['crop_disease_label']: sample['numeric_label'] for sample in self.test_set}
        self.idx_label = {v: k for k, v in self.label_idx.items()}
    
    def collate_fn(self, batch):
        images = [self.transform(sample['image'].convert('RGB')) for sample in batch]
        labels = [sample['numeric_label'] for sample in batch]
        return torch.stack(images), torch.tensor(labels)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, collate_fn=partial(agripath_collate, transform=self.transform), num_workers=8, persistent_workers=True, pin_memory=True, prefetch_factor=4,)
    
    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, collate_fn=partial(agripath_collate, transform=self.transform), num_workers=8, persistent_workers=True, pin_memory=True, prefetch_factor=4,)
    
    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, collate_fn=partial(agripath_collate, transform=self.transform), num_workers=8, persistent_workers=True, pin_memory=True, prefetch_factor=4,)
    
    def lab_loader(self):
        return DataLoader(self.lab_test, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=8, persistent_workers=True, pin_memory=True)
    
    def field_loader(self):
        return DataLoader(self.field_test, batch_size=self.batch_size, shuffle=False, collate_fn=self.collate_fn, num_workers=8, persistent_workers=True, pin_memory=True)
    
    def return_labels(self):
        return self.label_idx, self.idx_label

# A class that defines and modifies the ResNet50 model so that it can be used with the DataModule defined above
class CNNLightningModel(pl.LightningModule):
    def __init__(self, num_classes: int, backbone: models.ResNet|models.ConvNeXt, learning_rate=2e-4):
        super().__init__()
        
        # Log HyperParams
        self.save_hyperparameters(ignore=["backbone"])
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Load pre-trained ResNet50 model and freeze early layers for feature extraction
        self.backbone = backbone

        if isinstance(self.backbone, models.ResNet):
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Unfreeze last residual block for fine-tuning
            for p in self.backbone.layer4.parameters():
                p.requires_grad = True

            # Remove original Fully Connected Layer (optimised for ImageNet)
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
            # Create custom classification head
            self.classifier = nn.Linear(in_features, num_classes)

        elif isinstance(self.backbone, models.ConvNeXt):
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Unfreeze last residual block for fine-tuning
            for p in self.backbone.features[-1].parameters():
                p.requires_grad = True

            # Remove original Fully Connected Layer (optimised for ImageNet)
            in_features = self.backbone.classifier[2].in_features
            self.backbone.classifier[2] = nn.Identity()
            # Create custom classification head
            self.classifier = nn.Linear(in_features, num_classes)

        # Loss function and metrics
        self.criterion = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes=num_classes, average='macro')
    
    def forward(self, x):
        features = self.backbone(x) #Extract features
        out = self.classifier(features) #Final pass through custom classifier
        return out
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        out = self.forward(images)
        loss = self.criterion(out, labels)
        acc = self.accuracy(out, labels)

        self.log("train/loss", loss, on_epoch=True, prog_bar=True)
        self.log("train/acc", acc, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        out = self.forward(images)
        loss = self.criterion(out, labels)
        acc = self.accuracy(out, labels)

        self.log("val/loss", loss, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_epoch=True, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        out = self.forward(images)
        loss = self.criterion(out, labels)
        acc = self.accuracy(out, labels)
        
        self.log("test/loss", loss)
        self.log("test/acc", acc)

        return {'loss': loss, 'outputs': out, 'labels': labels}

    def configure_optimizers(self):
        return torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.learning_rate)

def check_loader(hf_repo: str):
    train_loader = AgriPathDataModule(hf_repo, batch_size=4)
    train_loader.setup()
    train_loader = train_loader.train_dataloader()

    images, labels = next(iter(train_loader))

    print(type(images), images.shape)
    print(type(labels), labels.shape)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--data', type=str, choices=['main', 'lab', 'field'], help="Type of HuggingFace Dataset (choose from main, lab, field)", default='main')
    parser.add_argument('-e', '--max_epochs', type=int, help="Number of epochs to run during training", default=10)
    parser.add_argument('-m', '--model', type=str, choices=['resnet50', 'convnext'], help="Select model to train (choose from resnet, convnext)", required=True)
    args = parser.parse_args()

    if args.data == 'main':
        hf_repo = "hamzamooraj99/AgriPath-LF16-30k"
    elif args.data == 'lab':
        hf_repo = "hamzamooraj99/AgriPath-LF16-30k-LAB"
    elif args.data == 'field':
        hf_repo = "hamzamooraj99/AgriPath-LF16-30k-FIELD"

    batch_sizes = [16, 32, 64]
    learning_rates = [1e-4, 5e-4, 2e-4]
    num_classes = 65
    max_epochs = args.max_epochs
    experiment_id = 0

    save_dir = f"{args.data}_{args.model}_experiments"
    os.makedirs(save_dir, exist_ok=True)

    seed_everything(42)

    run = wandb.init(
        project="AgriPath-VLM",
        job_type=f'{args.model}_{args.data}_ckpts',
        config={
            "model": args.model,
            "dataset": f"AgriPath-LF16-30k-{args.data.upper()}",
            "max_epochs": max_epochs,
            "num_classes": num_classes,
            "sweep_info": {
                "batch_sizes": batch_sizes,
                "learning_rates": learning_rates
            },
            "num_checkpoints": len(batch_sizes) * len(learning_rates),
        }
    )

    artifact = wandb.Artifact(
        name=f"{args.model}_{args.data}_checkpoints",
        type='model',
        description=f"Nine {args.model} CNN .pth checkpoints for {args.data} split",
        metadata=run.config.as_dict()
    )

    for batch_size, lr in itertools.product(batch_sizes, learning_rates):
        print(f"\n==== Running Experiment {experiment_id}: Batch Size = {batch_size}, LR = {lr} ====\n")

        # Define model with new learning rate
        print(f"\nDefine model with learning rate {lr}")
        the_model = None
        try:
            if args.model == "resnet50":
                the_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            elif args.model == "convnext":
                the_model = models.convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        except ValueError:
            print("INVALID MODEL")
        
        model = CNNLightningModel(num_classes=num_classes, learning_rate=lr, backbone=the_model)

        # Trainer setup
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator='gpu',
            devices=1,
            log_every_n_steps=10
        )

        # Load dataset with new batch_size
        print(f"Loading dataset with batch size {batch_size}")
        datamodule = AgriPathDataModule(hf_repo=hf_repo, batch_size=batch_size)

        # Train model
        print(f"\nTraining model")
        trainer.fit(model, datamodule=datamodule)

        # Test model
        print(f"\nTesting model")
        trainer.test(model, datamodule=datamodule)

        # Save model
        print(f"\nSaving model")
        ckpt_name = f"{args.model}_agripath_exp_{lr}_{batch_size}.pth"
        ckpt_path = os.path.join(save_dir, ckpt_name)
        torch.save(model.state_dict(), ckpt_path)

        artifact.add_file(local_path=ckpt_path, name=ckpt_name)

    run.log_artifact(artifact)