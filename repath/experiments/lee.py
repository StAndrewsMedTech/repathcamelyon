import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import torch
import random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.models import densenet121
from torchvision.transforms import Compose, ToTensor, RandomCrop, Normalize

from repath.utils.paths import project_root
from repath.data.datasets.dataset import Dataset
import repath.data.datasets.camelyon16 as camelyon16
import repath.data.datasets.camelyon17 as camelyon17
from repath.preprocess.tissue_detection import TissueDetectorGreyScale
from repath.preprocess.patching import GridPatchFinder, SlidesIndex
from repath.preprocess.sampling import split_camelyon16, split_camelyon17, balanced_sample
from repath.postprocess.results import SlidesIndexResults
from repath.postprocess.patch_level_results import patch_level_metrics
from repath.utils.seeds import set_seed
"""
Global stuff
"""
experiment_name = "lee"
experiment_root = project_root() / "experiments" / experiment_name
tissue_detector = TissueDetectorGreyScale()

global_seed = 123
"""

"""


class PatchClassifier(pl.LightningModule):
    def __init__(self, learnrate) -> None:
        super().__init__()
        model = densenet121(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1000, bias=True),
            nn.Linear(1000, 2))
        self.model = model
        self.learnrate = learnrate

    def step(self, batch, batch_idx, label):
        x, y = batch
        logits = self.model(x)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)
        self.log(f"{label}_loss", loss)

        pred = torch.log_softmax(logits, dim=1)
        correct = pred.argmax(dim=1).eq(y).sum().item()
        total = len(y)
        accu = correct / total
        self.log(f"{label}_accuracy", accu)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=self.learnrate,
                                    momentum=0.9,
                                    weight_decay=0.0001)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1),
            'interval': 'epoch'
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)


"""
Experiment step
"""


def preprocess_indexes() -> None:
    set_seed(global_seed)
    patch_finder = GridPatchFinder(labels_level=6, patch_level=0, patch_size=256, stride=256)

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
    set_seed(global_seed)
    # initalise datasets
    train_data16 = camelyon16.training()
    train_data17 = camelyon17.training()

    # load in the train and valid indexes
    train16 = SlidesIndex.load(train_data16, experiment_root / "train_index16")
    valid16 = SlidesIndex.load(train_data16, experiment_root / "valid_index16")
    train17 = SlidesIndex.load(train_data17, experiment_root / "train_index17")
    valid17 = SlidesIndex.load(train_data17, experiment_root / "valid_index17")

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
    set_seed(global_seed)
    # transforms
    transform = Compose([
        RandomCrop((240, 240)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # prepare our data
    batch_size = 64
    train_set = ImageFolder(experiment_root / "training_patches", transform=transform)
    valid_set = ImageFolder(experiment_root / "validation_patches", transform=transform)
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
    classifier = PatchClassifier(learnrate=0.1)
    trainer = pl.Trainer(callbacks=[checkpoint_callback], gpus=8, accelerator="ddp", max_epochs=15,
                         logger=csv_logger, log_every_n_steps=1)
    trainer.fit(classifier, train_dataloader=train_loader, val_dataloaders=valid_loader)


def inference_on_train() -> None:
    set_seed(global_seed)
    cp_path = list((experiment_root / "patch_model").glob("*.ckpt"))[0]
    classifier = PatchClassifier.load_from_checkpoint(checkpoint_path=cp_path)

    output_dir16 = experiment_root / "train_index16" / "pre_hnm_results"
    output_dir17 = experiment_root / "train_index17" / "pre_hnm_results"

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    train16 = SlidesIndex.load(camelyon16.training(), experiment_root / "train_index16")
    train17 = SlidesIndex.load(camelyon17.training(), experiment_root / "train_index17")

    transform = Compose([
        RandomCrop((240, 240)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_results16 = SlidesIndexResults.predict(train16, classifier, transform, 128, output_dir16,
                                                         results_dir_name, heatmap_dir_name)
    train_results16.save()

    train_results17 = SlidesIndexResults.predict(train17, classifier, transform, 128, output_dir17,
                                                         results_dir_name, heatmap_dir_name)
    train_results17.save()


def create_hnm_patches() -> None:
    set_seed(global_seed)
    input_dir16 = experiment_root / "train_index16" / "pre_hnm_results"
    input_dir17 = experiment_root / "train_index17" / "pre_hnm_results"

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    train_results16 = SlidesIndexResults.load_results_index(camelyon16.training(), input_dir16, results_dir_name,
                                                            heatmap_dir_name)
    train_results17 = SlidesIndexResults.load_results_index(camelyon17.training(), input_dir17, results_dir_name,
                                                            heatmap_dir_name)

    train_results = CombinedIndex.for_slide_indexes([train_results16, train_results17])

    FP_mask = np.logical_and(train_results.patches_df['tumor'] > 0.5, train_results.patches_df['label'] == 1)

    hnm_patches_df = train_results.patches_df[FP_mask]
    hnm_patches_df = hnm_patches_df.sort_values('tumor', axis=0, ascending=False)
    ### limit number of patches to same number as original patches
    hnm_patches_df = hnm_patches_df.iloc[0:47574]

    train_results.patches_df = hnm_patches_df

    train_results.save_patches(experiment_root / "training_patches", affix='-hnm')


def retrain_patch_classifier_hnm() -> None:
    set_seed(global_seed)
    # transforms
    transform = Compose([
        RandomCrop((240, 240)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # prepare our data
    batch_size = 64
    train_set = ImageFolder(experiment_root / "training_patches", transform=transform)
    valid_set = ImageFolder(experiment_root / "validation_patches", transform=transform)
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
    csv_logger = pl_loggers.CSVLogger(experiment_root / 'logs', name='patch_classifier_hnm', version=1)

    # train our model
    cp_path = list((experiment_root / "patch_model").glob("*.ckpt"))[0]
    classifier = PatchClassifier(learnrate=0.01).load_from_checkpoint(checkpoint_path=cp_path)
    trainer = pl.Trainer(callbacks=[checkpoint_callback], gpus=8, accelerator="ddp", max_epochs=15,
                         logger=csv_logger, log_every_n_steps=1)
    trainer.fit(classifier, train_dataloader=train_loader, val_dataloaders=valid_loader)
    

def inference_on_valid() -> None:
    set_seed(global_seed)
    cp_path = list((experiment_root / "patch_model_hnm").glob("*.ckpt"))[0]
    classifier = PatchClassifier.load_from_checkpoint(checkpoint_path=cp_path)

    output_dir16 = experiment_root / "post_hnm_results" / "valid16"
    output_dir17 = experiment_root / "post_hnm_results" / "valid17"

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    valid16 = SlidesIndex.load(camelyon16.training(), experiment_root / "valid_index16")
    valid17 = SlidesIndex.load(camelyon17.training(), experiment_root / "valid_index17")

    # valid16.patches = valid16[0:32]
    # valid17.patches = valid17[0:32]

    transform = Compose([
        RandomCrop((240, 240)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    valid_results16 = SlidesIndexResults.predict(valid16, classifier, transform, 128, output_dir16,
                                                         results_dir_name, heatmap_dir_name)
    valid_results16.save()

    valid_results17 = SlidesIndexResults.predict(valid17, classifier, transform, 128, output_dir17,
                                                         results_dir_name, heatmap_dir_name)
    valid_results17.save()


def preprocess_test_index() -> None:
    set_seed(global_seed)
    patch_finder = GridPatchFinder(labels_level=6, patch_level=0, patch_size=256, stride=256)

    # initalise datasets
    test_data16 = camelyon16.testing()
    test_data17 = camelyon17.testing()

    # find all patches in datasets
    test_patches16 = SlidesIndex.index_dataset(test_data16, tissue_detector, patch_finder)
    test_patches17 = SlidesIndex.index_dataset(test_data17, tissue_detector, patch_finder)

    # save
    test_patches16.save(experiment_root / "test_index16")
    test_patches17.save(experiment_root / "test_index17")



def inference_on_test() -> None:
    set_seed(global_seed)
    cp_path = list((experiment_root / "patch_model_hnm").glob("*.ckpt"))[0]
    classifier = PatchClassifier.load_from_checkpoint(checkpoint_path=cp_path)

    output_dir16 = experiment_root / "post_hnm_results" / "test16"
    output_dir17 = experiment_root / "post_hnm_results" / "test17"

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    test16 = SlidesIndex.load(camelyon16.testing(), experiment_root / "test_index16")
    test17 = SlidesIndex.load(camelyon17.testing(), experiment_root / "test_index17")

    # valid16.patches = valid16[0:32]
    # valid17.patches = valid17[0:32]

    transform = Compose([
        RandomCrop((240, 240)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_results16 = SlidesIndexResults.predict(test16, classifier, transform, 128, output_dir16,
                                                         results_dir_name, heatmap_dir_name)
    test_results16.save()

    test_results17 = SlidesIndexResults.predict(test17, classifier, transform, 128, output_dir17,
                                                         results_dir_name, heatmap_dir_name)
    test_results17.save()


def calculate_patch_level_results() -> None:
    def patch_dataset_function(modelname: str, splitname: str, dataset16: SlideDataset, dataset17: Dataset, ci: bool = False):
        # define strings for model and split
        model_dir_name = modelname + '_results'
        splitname16 = splitname + '16'
        splitname17 = splitname + '17'
        splitname1617 = splitname + '1617'
        results_out_name = "patch_summaries"
        results_in_name = "results"
        heatmap_in_name = "heatmaps"

        # set paths for model and split
        model_dir = experiment_root / model_dir_name
        splitdirin16 = model_dir / splitname16
        splitdirin17 = model_dir / splitname17
        splitdirout16 = model_dir / results_out_name / splitname16
        splitdirout17 = model_dir / results_out_name / splitname17
        splitdirout1617 = model_dir / results_out_name / splitname1617

        # read in predictions
        split_results_16 = SlidesIndexResults.load_results_index(dataset16, splitdirin16,
                                                                 results_in_name, heatmap_in_name)
        split_results_17 = SlidesIndexResults.load_results_index(dataset17, splitdirin17,
                                                                 results_in_name, heatmap_in_name)
        split_results_17_annotated = split_results_17.select_annotated()

        # calculate patch level results
        title16 = experiment_name + ' experiment ' + modelname + ' model Camelyon 16 ' + splitname + ' dataset'
        patch_level_metrics([split_results_16], splitdirout16, title16, ci=False)
        title17 = experiment_name + ' experiment ' + modelname + ' model Camelyon 17 ' + splitname + ' dataset'
        patch_level_metrics([split_results_17], splitdirout17, title17, ci=False)
        title1617 = experiment_name + ' experiment ' + modelname + ' model Camelyon 16 & 17 ' + splitname + ' dataset'
        patch_level_metrics([split_results_1617], splitdirout1617, title1617, ci=False)

    set_seed(global_seed)

    patch_dataset_function("pre_hnm", "valid", camelyon16.training(), camelyon17.training(), ci=True)
    patch_dataset_function("pre_hnm", "test", camelyon16.testing(), camelyon17.testing(), ci=True)
    patch_dataset_function("post_hnm", "valid", camelyon16.training(), camelyon17.training(), ci=True)
    patch_dataset_function("post_hnm", "test", camelyon16.testing(), camelyon17.testing(), ci=True)


### Temp for debugging
def calculate_patch_level_results_valid16() -> None:
    def patch_dataset_function(modelname: str, splitname: str, dataset16: Dataset, ci: bool = False):
        # define strings for model and split
        model_dir_name = modelname + '_results'
        splitname16 = splitname + '16'
        results_out_name = "patch_summaries"
        results_in_name = "results"
        heatmap_in_name = "heatmaps"

        # set paths for model and split
        model_dir = experiment_root / model_dir_name
        splitdirin16 = model_dir / splitname16
        splitdirout16 = model_dir / results_out_name / splitname16

        # read in predictions
        split_results_16 = SlidesIndexResults.load(dataset16, splitdirin16,
                                                                 results_in_name, heatmap_in_name)

        # calculate patch level results
        title16 = experiment_name + ' experiment ' + modelname + ' model Camelyon 16 ' + splitname + ' dataset'
        patch_level_metrics([split_results_16], splitdirout16, title16, ci=False, nreps=10)

    set_seed(global_seed)

    patch_dataset_function("post_hnm", "valid", camelyon16.training(), ci=True)
