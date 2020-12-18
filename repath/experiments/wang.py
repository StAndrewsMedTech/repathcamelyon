import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomRotation, Normalize

from repath.utils.paths import project_root
import repath.data.datasets.camelyon16 as camelyon16
from repath.preprocess.tissue_detection import TissueDetectorOTSU
from repath.preprocess.patching import GridPatchFinder, SlidesIndex
from repath.preprocess.sampling import split_camelyon16, balanced_sample
from torchvision.models import googlenet

"""
Global stuff
"""
experiment_name = "wang"
experiment_root = project_root() / "experiments" / experiment_name
tissue_detector = TissueDetectorOTSU()

class PatchClassifier(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = googlenet(num_classes=2)

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
                                    lr=0.01, 
                                    momentum=0.9, 
                                    weight_decay=0.0005)
        scheduler = {
            'scheduler': torch.optim.StepLR(optimizer, step_size=50000, gamma=0.5),
            'interval': 'step' 
        }
        return optimizer, scheduler

"""
Experiment step
"""
def preprocess_indexes() -> None:
    # index all the patches for the camelyon16 dataset
    train_data = camelyon16.training()
    patch_finder = GridPatchFinder(labels_level=5, patch_level=0, patch_size=256, stride=32)
    train_patches = SlidesIndex.index_dataset(train_data, tissue_detector, patch_finder)

    # do the train validate split
    train, valid = split_camelyon16(train_patches, 0.8)
    train.save(experiment_root / "train_index")
    valid.save(experiment_root / "valid_index")


def preprocess_samples() -> None:
    # load in the train and valid indexes
    train = SlidesIndex.load(experiment_root / "train_index")
    valid = SlidesIndex.load(experiment_root / "valid_index")

    # sample from train and valid sets
    train_samples = balanced_sample([train], 2000000)
    valid_samples = balanced_sample([valid], 500000)

    # save out all the patches
    train_samples.save_patches(experiment_root / "training_patches")
    valid_samples.save_patches(experiment_root / "validation_patches")


def train_patch_classifier() -> None:
    # prepare our data
    batch_size = 32
    train_set = ImageFolder.load(experiment_root / "training_patches")
    valid_set = ImageFolder.load(experiment_root / "validation_patches")
    
    # transforms
    transform = Compose([
        RandomRotation((0, 360)),
        RandomCrop((224, 224)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # create dataloaders
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
