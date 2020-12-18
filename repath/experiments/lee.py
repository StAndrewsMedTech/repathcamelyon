import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.Functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import densenet121
from torchvision.transforms import Compose, ToTensor, RandomCrop, Normalize

from repath.utils.paths import project_root
import repath.data.datasets.camelyon16 as camelyon16
import repath.data.datasets.camelyon17 as camelyon17
from repath.preprocess.tissue_detection import TissueDetectorGreyScale
from repath.preprocess.patching import GridPatchFinder, SlidesIndex
from repath.preprocess.sampling import split_camelyon16, split_camelyon17, balanced_sample, select_annotated

"""
Global stuff
"""
experiment_name = "lee"
experiment_root = project_root() / "experiments" / experiment_name
tissue_detector = TissueDetectorGreyScale()

"""

"""

class PatchClassifier(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        model = densenet121(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1000, bias=True),
            nn.Linear(1000, 2))
        self.model = model

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def step(self, batch, batch_idx, label):
        x, y = batch
        logits = self.model(x)
        pred = torch.log_softmax(logits, dim=1)
        loss = self.cross_entropy_loss(logits, y)
        self.log(f"{label}_loss", loss)
        
        correct=pred.argmax(dim=1).eq(y).sum().item()
        total=len(y)   
        accu = correct / total
        self.log(f"{label}_accuracy", accu)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), 
                                    lr=0.1, 
                                    momentum=0.9, 
                                    weight_decay=1e-4)           
        scheduler = {
            'scheduler': torch.optim.StepLR(optimizer, step_size=10, gamma=0.1),
            'interval': 'epoch' 
        }
        return optimizer, scheduler

"""
Experiment step
"""
def preprocess_indexes() -> None:
    patch_finder = GridPatchFinder(labels_level=8, patch_level=0, patch_size=256, stride=256)

    # initalise datasets
    train_data16 = camelyon16.training()
    train_data17 = camelyon17.training()

    # find all patches in datasets
    train_patches16 = SlidesIndex.index_dataset(train_data16, tissue_detector, patch_finder)
    train_patches17 = SlidesIndex.index_dataset(train_data17, tissue_detector, patch_finder)

    # do the train validate split for camelyon16
    train16, valid16 = split_camelyon16(train_patches16, 0.62)
    train16.save(experiment_root / "train_index16")
    valid16.save(experiment_root / "valid_index16")

    # do the train validate split for camelyon17
    train17, valid17 = split_camelyon17(train_patches17, 0.62)
    train17.save(experiment_root / "train_index17")
    valid17.save(experiment_root / "valid_index17")


def preprocess_samples() -> None:
    # load in the train and valid indexes
    train16 = SlidesIndex.load(experiment_root / "train_index16")
    valid16 = SlidesIndex.load(experiment_root / "valid_index16")
    train17 = SlidesIndex.load(experiment_root / "train_index17")
    valid17 = SlidesIndex.load(experiment_root / "valid_index17")

    # remove non-annotated slides from camelyon17
    train17 = select_annotated(train17)
    valid17 = select_annotated(valid17)

    # sample from train and valid sets
    train_samples = balanced_sample([train16, train17], 47574)
    valid_samples = balanced_sample([valid16, valid17], 29000)

    # save out all the patches
    train_samples.save_patches(experiment_root / "training_patches")
    valid_samples.save_patches(experiment_root / "validation_patches")


def train_patch_classifier() -> None:
    # transforms
    transform = Compose([
        RandomCrop((240, 240)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # prepare our data
    batch_size = 64
    train_set = ImageFolder.load(experiment_root / "training_patches")
    valid_set = ImageFolder.load(experiment_root / "validation_patches")
    train_loader = DataLoader(train_set, batch_size=batch_size, transform=transform, num_workers=80)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, transform=transform, num_workers=80)

    # configure logging and checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=experiment_root / "patch_model",
        filename=f"checkpoint.ckpt",
        save_top_k=1,
        mode="min",
    )

    # train our model
    classifier = PatchClassifier()
    trainer = pl.Trainer(callbacks=[checkpoint_callback])
    trainer.fit(classifier, train_dataloader=train_loader, val_dataloaders=valid_loader)
