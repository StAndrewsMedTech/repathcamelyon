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
from repath.preprocess.patching import GridPatchFinder, SlidesIndex, CombinedIndex
from repath.preprocess.sampling import split_camelyon16, balanced_sample
from repath.postprocess.results import SlidesIndexResults
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

    train_overlapping = SlidesIndex.load(camelyon16.training(), experiment_root / "train_index")

    # original training data indexes are overlapping we need non overlapping grid for inference
    train_ol_slides = [pat.slide_path for pat in train_overlapping.patches]
    train_data_cut_down = camelyon16.training()
    mask = [sl in train_ol_slides for sl in train_data_cut_down.paths.slide]
    train_data_cut_down.paths = train_data_cut_down.paths[mask]

    #patch_finder = GridPatchFinder(labels_level=5, patch_level=0, patch_size=256, stride=256)
    #train_patches_grid = SlidesIndex.index_dataset(train_data_cut_down, tissue_detector, patch_finder)
    #train_patches_grid.save(experiment_root / "train_index_grid")

    train_patches_grid = SlidesIndex.load(train_data_cut_down, experiment_root / "train_index_grid")
    # using only the grid patches finds only 340 false positives, not enough to retrain so try using overlapping to create more patches

    transform = Compose([
        RandomCrop((224, 224)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_results16 = SlidesIndexResults.predict(train_patches_grid, classifier, transform, 128, output_dir16,
                                                 results_dir_name, heatmap_dir_name, nthreads=2)
    train_results16.save()


def create_hnm_patches() -> None:
    set_seed(global_seed)
    input_dir16 = experiment_root / "pre_hnm_results" / "train16"

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    train_results16 = SlidesIndexResults.load(camelyon16.training(), input_dir16, results_dir_name, heatmap_dir_name)

    train_results = CombinedIndex.for_slide_indexes([train_results16])

    FP_mask = np.logical_and(train_results.patches_df['tumor'] > 0.5, train_results.patches_df['label'] == 1)

    hnm_patches_df = train_results.patches_df[FP_mask]
    hnm_patches_df = hnm_patches_df.sort_values('tumor', axis=0, ascending=False)
    ### limit number of patches to same number as original patches
    n_hnm_patches = min(2000000, hnm_patches_df.shape[0])
    hnm_patches_df = hnm_patches_df.iloc[0:n_hnm_patches]
    print(n_hnm_patches)

    train_results.patches_df = hnm_patches_df

    train_results.save_patches(experiment_root / "training_patches", affix='-hnm', add_patches=True)


def retrain_patch_classifier_hnm() -> None:
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
        dirpath=experiment_root / "patch_model_hnm",
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
    csv_logger = pl_loggers.CSVLogger(experiment_root / 'logs', name='patch_classifier_hnm', version=0)

    # train our model
    cp_path = list((experiment_root / "patch_model").glob("*.ckpt"))[0]
    classifier = PatchClassifier.load_from_checkpoint(checkpoint_path=cp_path)
    trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stop_callback], gpus=8, accelerator="ddp", max_epochs=15, 
                     logger=csv_logger, log_every_n_steps=1)
    trainer.fit(classifier, train_dataloader=train_loader, val_dataloaders=valid_loader)


def inference_on_valid_pre() -> None:
    set_seed(global_seed)
    cp_path = list((experiment_root / "patch_model").glob("*.ckpt"))[0]
    classifier = PatchClassifier.load_from_checkpoint(checkpoint_path=cp_path)

    output_dir16 = experiment_root / "pre_hnm_results" / "valid16"

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    valid_overlapping = SlidesIndex.load(camelyon16.training(), experiment_root / "valid_index")

    # original training data indexes are overlapping we need non overlapping grid for inference
    valid_ol_slides = [pat.slide_path for pat in valid_overlapping.patches]
    valid_data_cut_down = camelyon16.training()
    mask = [sl in valid_ol_slides for sl in valid_data_cut_down.paths.slide]
    valid_data_cut_down.paths = valid_data_cut_down.paths[mask]

    patch_finder = GridPatchFinder(labels_level=5, patch_level=0, patch_size=256, stride=256)
    valid_patches_grid = SlidesIndex.index_dataset(valid_data_cut_down, tissue_detector, patch_finder)
    valid_patches_grid.save(experiment_root / "valid_index_grid")

    valid_patches_grid = SlidesIndex.load(valid_data_cut_down, experiment_root / "valid_index_grid")
    # using only the grid patches finds only 340 false positives, not enough to retrain so try using overlapping to create more patches

    transform = Compose([
        RandomCrop((224, 224)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    valid_results16 = SlidesIndexResults.predict(valid_patches_grid, classifier, transform, 128, output_dir16,
                                                 results_dir_name, heatmap_dir_name, nthreads=2)
    valid_results16.save()


def inference_on_valid_post() -> None:
    set_seed(global_seed)
    cp_path = list((experiment_root / "patch_model_hnm").glob("*.ckpt"))[0]
    classifier = PatchClassifier.load_from_checkpoint(checkpoint_path=cp_path)

    output_dir16 = experiment_root / "post_hnm_results" / "valid16"

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    valid_overlapping = SlidesIndex.load(camelyon16.training(), experiment_root / "valid_index")

    # original training data indexes are overlapping we need non overlapping grid for inference
    valid_ol_slides = [pat.slide_path for pat in valid_overlapping.patches]
    valid_data_cut_down = camelyon16.training()
    mask = [sl in valid_ol_slides for sl in valid_data_cut_down.paths.slide]
    valid_data_cut_down.paths = valid_data_cut_down.paths[mask]

    valid_patches_grid = SlidesIndex.load(valid_data_cut_down, experiment_root / "valid_index_grid")
    # using only the grid patches finds only 340 false positives, not enough to retrain so try using overlapping to create more patches

    transform = Compose([
        RandomCrop((224, 224)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    valid_results16 = SlidesIndexResults.predict(valid_patches_grid, classifier, transform, 128, output_dir16,
                                                 results_dir_name, heatmap_dir_name, nthreads=2)
    valid_results16.save()


def preprocess_test_index() -> None:
    set_seed(global_seed)
    patch_finder = GridPatchFinder(labels_level=5, patch_level=0, patch_size=256, stride=256)

    # initalise datasets
    test_data16 = camelyon16.testing()

    # find all patches in datasets
    test_patches16 = SlidesIndex.index_dataset(test_data16, tissue_detector, patch_finder)

    # save
    test_patches16.save(experiment_root / "test_index16")


def inference_on_test_pre() -> None:
    set_seed(global_seed)
    cp_path = list((experiment_root / "patch_model").glob("*.ckpt"))[0]
    classifier = PatchClassifier.load_from_checkpoint(checkpoint_path=cp_path)

    output_dir16 = experiment_root / "pre_hnm_results" / "test16"

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    test_patches = SlidesIndex.load(camelyon16.testing(), experiment_root / "test_index16")
    # using only the grid patches finds only 340 false positives, not enough to retrain so try using overlapping to create more patches

    transform = Compose([
        RandomCrop((224, 224)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_results16 = SlidesIndexResults.predict(test_patches, classifier, transform, 128, output_dir16,
                                                 results_dir_name, heatmap_dir_name, nthreads=2)
    test_results16.save()


def inference_on_test_post() -> None:
    set_seed(global_seed)
    cp_path = list((experiment_root / "patch_model_hnm").glob("*.ckpt"))[0]
    classifier = PatchClassifier.load_from_checkpoint(checkpoint_path=cp_path)

    output_dir16 = experiment_root / "post_hnm_results" / "test16"

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    test_patches = SlidesIndex.load(camelyon16.testing(), experiment_root / "test_index16")
    # using only the grid patches finds only 340 false positives, not enough to retrain so try using overlapping to create more patches

    transform = Compose([
        RandomCrop((224, 224)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_results16 = SlidesIndexResults.predict(test_patches, classifier, transform, 128, output_dir16,
                                                 results_dir_name, heatmap_dir_name, nthreads=2)
    test_results16.save()

