import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomRotation, Normalize

from repath.utils.paths import project_root
import repath.data.datasets.camelyon16 as camelyon16
from repath.preprocess.tissue_detection import TissueDetectorOTSU
from repath.preprocess.patching import GridPatchFinder, SlidesIndex
from repath.preprocess.sampling import split_camelyon16, balanced_sample
from torchvision.models import GoogLeNet
from repath.utils.seeds import set_seed

"""
Global stuff
"""
experiment_name = "wang"
experiment_root = project_root() / "experiments" / experiment_name
tissue_detector = TissueDetectorOTSU()

global_seed = 123


class PatchClassifier(pl.LightningModule):
    
    def __init__(self) -> None:
        super().__init__()
        self.model = GoogLeNet(num_classes=2)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output, aux2, aux1 = self.model(x)
        pred = torch.log_softmax(output, dim=1)

        criterion = nn.CrossEntropyLoss()
        loss1 = criterion(output, y)
        loss2 = criterion(aux1, y)
        loss3 = criterion(aux2, y)
        loss = loss1 + 0.3 * loss2 + 0.3 * loss3
        self.log("train_loss", loss)
        
        correct=pred.argmax(dim=1).eq(y).sum().item()
        total=len(y)   
        accu = correct / total
        self.log("train_accuracy", accu)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        pred = torch.log_softmax(output, dim=1)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y)
        self.log("val_loss", loss)
        
        correct=pred.argmax(dim=1).eq(y).sum().item()
        total=len(y)   
        accu = correct / total
        self.log("val_accuracy", accu)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), 
                                    lr=0.01, 
                                    momentum=0.9, 
                                    weight_decay=0.0005)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.5),
            'interval': 'step' 
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

"""
Experiment step
"""
def preprocess_indexes() -> None:
    """ Generates patch_index files for train and validation slide
    """
    set_seed(global_seed)
    # index all the patches for the camelyon16 dataset
    train_data = camelyon16.training()
    patch_finder = GridPatchFinder(labels_level=5, patch_level=0, patch_size=256, stride=32)
    train_patches = SlidesIndex.index_dataset(train_data, tissue_detector, patch_finder)

    # do the train validate split
    train, valid = split_camelyon16(train_patches, 0.8)
    train.save(experiment_root / "train_index")
    valid.save(experiment_root / "valid_index")


def preprocess_samples() -> None:
    """Generates all patches for train and validation sets
    """
    set_seed(global_seed)
    # load in the train and valid indexes
    train_data = camelyon16.training()
    train = SlidesIndex.load(train_data, experiment_root / "train_index")
    valid = SlidesIndex.load(train_data, experiment_root / "valid_index")

    # sample from train and valid sets
    train_samples = balanced_sample([train], 2000000)
    valid_samples = balanced_sample([valid], 500000)

    # save out all the patches
    train_samples.save_patches(experiment_root / "training_patches")
    valid_samples.save_patches(experiment_root / "validation_patches")


def train_patch_classifier() -> None:
    """ Trains a classifier on the train patches and validates on validation patches.
    """
    set_seed(global_seed)
    # transforms
    transform = Compose([
        RandomRotation((0, 360)),
        RandomCrop((224, 224)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # prepare our data
    batch_size = 32
    train_set = ImageFolder(experiment_root / "training_patches", transform=transform)
    valid_set = ImageFolder(experiment_root / "validation_patches", transform=transform)   
    
    # create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, worker_init_fn=np.random.seed(global_seed))
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=8, worker_init_fn=np.random.seed(global_seed))

    # configure logging and checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy",
        dirpath=experiment_root / "patch_model",
        filename=f"checkpoint.ckpt",
        save_top_k=1,
        mode="max",
    )

    early_stop_callback = EarlyStopping(
        monitor='val_accuracy',
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode='max'
    )

    # create a logger
    csv_logger = pl_loggers.CSVLogger(experiment_root / 'logs', name='patch_classifier', version=0)

    # train our model
    classifier = PatchClassifier()
    trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stop_callback], gpus=8, accelerator="ddp", max_epochs=15, 
                     logger=csv_logger, log_every_n_steps=1)
    trainer.fit(classifier, train_dataloader=train_loader, val_dataloaders=valid_loader)


def inference_on_train() -> None:
    set_seed(global_seed)
    cp_path = list((experiment_root / "patch_model").glob("*.ckpt"))[0]
    classifier = PatchClassifier.load_from_checkpoint(checkpoint_path=cp_path)

    output_dir16 = experiment_root / "pre_hnm_results" / "train16"

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    train16 = SlidesIndex.load(camelyon16.training(), experiment_root / "train_index")

    transform = Compose([
        RandomCrop((240, 240)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    #valid_results16 = SlidesIndexResults.predict(valid16, classifier, transform, 128, output_dir16,
    #                                                     results_dir_name, heatmap_dir_name)
    #valid_results16.save()
    print("Valid17")
    valid_results17 = SlidesIndexResults.predict(valid17, classifier, transform, 128, output_dir17,
                                                         results_dir_name, heatmap_dir_name)
    valid_results17.save()
