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
import repath.data.datasets.cervical_debug as cervical
from repath.data.datasets.dataset import Dataset
from repath.preprocess.tissue_detection import TissueDetectorGreyScale
from repath.preprocess.patching import GridPatchFinder, SlidesIndex, CombinedIndex
from repath.preprocess.sampling import split_cervical_tags, balanced_sample, get_subset_of_dataset, simple_random_replacement
from repath.postprocess.results import SlidesIndexResults
from torchvision.models import GoogLeNet
from repath.utils.seeds import set_seed
from repath.postprocess.patch_level_results import patch_level_metrics_multi


"""
Global stuff
"""
experiment_name = "cervical4"
experiment_root = project_root() / "experiments" / experiment_name
tissue_detector = TissueDetectorGreyScale()

global_seed = 123


class PatchClassifier(pl.LightningModule):
    
    def __init__(self) -> None:
        super().__init__()
        self.model = GoogLeNet(num_classes=4)

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
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5),
            'interval': 'epoch' 
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)


def preprocess_indexes() -> None:
    set_seed(global_seed)
    # index all the patches for the camelyon16 dataset
    train_data = cervical.debug()
    patch_finder = GridPatchFinder(labels_level=5, patch_level=0, patch_size=128, stride=128)
    train_patches = SlidesIndex.index_dataset(train_data, tissue_detector, patch_finder)
    
    train, valid = split_cervical_tags(train_patches)

    train.save(experiment_root / "train_index")
    valid.save(experiment_root / "valid_index")


def preprocess_samples() -> None:
    set_seed(global_seed)
    # initalise datasets
    train_data = cervical.debug()

    # load in the train and valid indexes
    train = SlidesIndex.load(train_data, experiment_root / "train_index")
    valid = SlidesIndex.load(train_data, experiment_root / "valid_index")

    # sample from train and valid sets
    train_samples = balanced_sample([train], 100000, floor_samples=50000)
    valid_samples = balanced_sample([valid], 50000, floor_samples=25000)

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
        #RandomCrop((224, 224)),
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
    trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stop_callback], gpus=8, accelerator="ddp", max_epochs=20, 
                     logger=csv_logger, log_every_n_steps=1)
    trainer.fit(classifier, train_dataloader=train_loader, val_dataloaders=valid_loader)


def inference_on_valid_pre() -> None:
    set_seed(global_seed)
    cp_path = list((experiment_root / "patch_model").glob("*.ckpt"))[0]
    classifier = PatchClassifier.load_from_checkpoint(checkpoint_path=cp_path)

    output_dir = experiment_root / "patch_results" / "valid"

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    valid_patches = SlidesIndex.load(cervical.debug(), experiment_root / "valid_index")

    transform = Compose([
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    valid_results = SlidesIndexResults.predict(valid_patches, classifier, transform, 128, output_dir,
                                                 results_dir_name, heatmap_dir_name, nthreads=2, 
                                                 heatmap_classes=['normal', 'low_grade', 'high_grade', 'malignant'])
    valid_results.save()


def calculate_patch_level_results() -> None:

    set_seed(global_seed)

    dirin = experiment_root / 'patch_results' / 'valid'
    dirout = experiment_root / 'patch_results' / "patch_summaries" / 'valid'

    val_results = SlidesIndexResults.load(cervical.debug(), dirin, "results", "heatmaps")

    title = experiment_name + ' experiment ' + 'patch' + ' model Cervical ' + 'valid' + ' dataset'

    patch_level_metrics_multi([val_results], dirout, title)

