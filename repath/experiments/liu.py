from repath.preprocess.patching.apply_transform import LiuTransform, MultiTransform
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
import torch
import numpy as np
import random
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor, RandomCrop, RandomRotation, Normalize, ColorJitter
from torchvision.models import inception_v3


from repath.data.datasets.dataset import Dataset
import repath.data.datasets.camelyon16 as camelyon16
from repath.preprocess.tissue_detection import TissueDetectorOTSU
from repath.preprocess.patching import GridPatchFinder, SlidesIndex
from repath.preprocess.sampling import split_camelyon16, balanced_sample, weighted_random, get_subset_of_dataset
from repath.preprocess.augmentation.augments import Rotate, FlipRotate
from repath.postprocess.patch_level_results import patch_level_metrics
from repath.postprocess.results import SlidesIndexResults
from repath.postprocess.slide_level_metrics import SlideClassifierLiu
from repath.utils.convert import average_patches
from repath.utils.paths import project_root
from repath.utils.seeds import set_seed


"""
Global stuff
"""
experiment_name = "liu"
experiment_root = project_root() / "experiments" / experiment_name
tissue_detector = TissueDetectorOTSU()
patch_border = 171
patch_jitter = 8

global_seed = 123


class PatchClassifier(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.model = inception_v3(num_classes=2)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output, aux1 = self.model(x)
        pred = torch.log_softmax(output, dim=1)

        criterion = nn.CrossEntropyLoss()
        loss1 = criterion(output, y)
        loss2 = criterion(aux1, y)
        ### TODO:  check how losses are weighted
        loss = loss1 + 0.4 * loss2
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
        optimizer = torch.optim.RMSprop(self.model.parameters(), 
                                    lr=0.05, 
                                    momentum=0.9, 
                                    weight_decay=0.0,
                                    alpha=0.9,
                                    eps= 1.0,
                                    centered= False)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.StepLR(optimizer, step_size=62500, gamma=0.5),
            'interval': 'step' 
        }
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.model(x)

"""
Experiment step
"""
def preprocess_indexes() -> None:
    set_seed(global_seed)
    # index all the patches for the camelyon16 dataset
    train_data = camelyon16.training()
    apply_transforms = LiuTransform(label=2, num_transforms=8)
    patch_finder = GridPatchFinder(labels_level=6, patch_level=0, patch_size=128, stride=128, 
                                   border=patch_border, jitter=patch_jitter, 
                                   apply_transforms=apply_transforms)
    train_patches = SlidesIndex.index_dataset(train_data, tissue_detector, patch_finder)

    # do the train validate split
    train, valid = split_camelyon16(train_patches, 0.8)
    train.save(experiment_root / "train_index")
    valid.save(experiment_root / "valid_index")


def preprocess_samples() -> None:
    set_seed(global_seed)
    # load in the train and valid indexes
    train_data = camelyon16.training()
    train = SlidesIndex.load(train_data, experiment_root / "train_index")
    valid = SlidesIndex.load(train_data, experiment_root / "valid_index")

    # sample from train and valid sets
    train_samples = balanced_sample([train], 5000000, sampling_policy=weighted_random)
    valid_samples = balanced_sample([valid], 1250000, sampling_policy=weighted_random)

    train_samples.save(experiment_root / "train_samples")
    valid_samples.save(experiment_root / "valid_samples")

    # create list of augmentations 
    augmentations = [Rotate(angle=0), Rotate(angle=90), Rotate(angle=180), Rotate(angle=270),
                     FlipRotate(angle=0), FlipRotate(angle=90), FlipRotate(angle=180), FlipRotate(angle=270)]

    # save out all the patches
    train_samples.save_patches(experiment_root / "training_patches", augmentations)
    valid_samples.save_patches(experiment_root / "validation_patches", augmentations)


def train_patch_classifier() -> None:
    set_seed(global_seed)
    # transforms
    transform = Compose([
        RandomCrop((299, 299)),
        ColorJitter(brightness=0.25, contrast=0.75, saturation=0.25, hue=0.04),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # prepare our data
    batch_size = 32
    train_set = ImageFolder(experiment_root / "training_patches", transform=transform)
    valid_set = ImageFolder(experiment_root / "validation_patches", transform=transform)
    
    # create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, worker_init_fn=np.random.seed(global_seed))
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=4, worker_init_fn=np.random.seed(global_seed))

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


def restart_patch_classifier() -> None:
    set_seed(global_seed)
    # transforms
    transform = Compose([
        RandomCrop((299, 299)),
        ColorJitter(brightness=0.25, contrast=0.75, saturation=0.25, hue=0.04),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # prepare our data
    batch_size = 32
    train_set = ImageFolder(experiment_root / "training_patches", transform=transform)
    valid_set = ImageFolder(experiment_root / "validation_patches", transform=transform)
    
    # create dataloaders
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=4, worker_init_fn=np.random.seed(global_seed))
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=4, worker_init_fn=np.random.seed(global_seed))

    # configure logging and checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy",
        dirpath=experiment_root / "patch_model",
        filename=f"checkpoint_restart",
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
    csv_logger = pl_loggers.CSVLogger(experiment_root / 'logs', name='patch_classifier', version=1)

    # train our model
    cp_path = experiment_root / "patch_model"/ "checkpoint.ckpt.ckpt"
    classifier = PatchClassifier.load_from_checkpoint(checkpoint_path=cp_path)
    trainer = pl.Trainer(callbacks=[checkpoint_callback, early_stop_callback], gpus=8, accelerator="ddp", max_epochs=15, 
                     logger=csv_logger, log_every_n_steps=1)
    trainer.fit(classifier, train_dataloader=train_loader, val_dataloaders=valid_loader)


def preprocess_valid_inference():
    set_seed(global_seed)
    # cp_path = list((experiment_root / "patch_model").glob("*.ckpt"))[0]
    valid = SlidesIndex.load(camelyon16.training(), experiment_root / "valid_index")

    # original validation data indexes only apply transforms to the tumor patches
    # for inference we want to apply transforms to all patches
    # get valid slides in valid slide index (the list of split between valid and train)
    valid_slides = [pat.slide_path for pat in valid.patches]
    # create new dataset initially with both training and valid
    valid_data_cut_down = camelyon16.training()
    # chop new dataset down to be just slide in valid set
    mask = [sl in valid_slides for sl in valid_data_cut_down.paths.slide]
    valid_data_cut_down.paths = valid_data_cut_down.paths[mask]

    # apply transforms for all 
    apply_transforms = MultiTransform(num_transforms=8)
    patch_finder = GridPatchFinder(labels_level=6, patch_level=0, patch_size=128, stride=128, 
                                   border=patch_border, jitter=patch_jitter, 
                                   apply_transforms=apply_transforms)
    valid_patches_8transforms = SlidesIndex.index_dataset(valid_data_cut_down, tissue_detector, patch_finder)
    valid_patches_8transforms.save(experiment_root / "valid_index_8transforms")


def inference_on_valid16() -> None:
    set_seed(global_seed)
    # cp_path = list((experiment_root / "patch_model").glob("*.ckpt"))[0]
    cp_path = experiment_root / "patch_model" / "checkpoint.ckpt-v0.ckpt"
    classifier = PatchClassifier.load_from_checkpoint(checkpoint_path=cp_path)

    output_dir16 = experiment_root / "inference_results" / "valid16"

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    # cp_path = list((experiment_root / "patch_model").glob("*.ckpt"))[0]
    valid = SlidesIndex.load(camelyon16.training(), experiment_root / "valid_index")

    # original validation data indexes only apply transforms to the tumor patches
    # for inference we want to apply transforms to all patches
    # get valid slides in valid slide index (the list of split between valid and train)
    valid_slides = [pat.slide_path for pat in valid.patches]
    # create new dataset initially with both training and valid
    valid_data_cut_down = camelyon16.training()
    # chop new dataset down to be just slide in valid set
    mask = [sl in valid_slides for sl in valid_data_cut_down.paths.slide]
    valid_data_cut_down.paths = valid_data_cut_down.paths[mask]

    valid_patches_8transforms = SlidesIndex.load(valid_data_cut_down, experiment_root / "valid_index_8transforms")

    transform = Compose([
        RandomCrop((299, 299)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # create list of augmentations 
    augmentations = [Rotate(angle=0), Rotate(angle=90), Rotate(angle=180), Rotate(angle=270),
                     FlipRotate(angle=0), FlipRotate(angle=90), FlipRotate(angle=180), FlipRotate(angle=270)]

    valid_results16 = SlidesIndexResults.predict(valid_patches_8transforms, classifier, transform, 128, output_dir16,
                                                 results_dir_name, heatmap_dir_name, augments=augmentations, nthreads=2)
    valid_results16.save()


def preprocess_testindex() -> None:
    set_seed(global_seed)
    # index all the patches for the camelyon16 dataset
    test_data = camelyon16.testing()
    # apply transforms for all 
    apply_transforms = MultiTransform(num_transforms=8)
    patch_finder = GridPatchFinder(labels_level=6, patch_level=0, patch_size=128, stride=128, 
                                   border=patch_border, jitter=patch_jitter, 
                                   apply_transforms=apply_transforms)
    test_patches = SlidesIndex.index_dataset(test_data, tissue_detector, patch_finder)

    test_patches.save(experiment_root / "test_index")


def inference_on_test16() -> None:
    set_seed(global_seed)
    # cp_path = list((experiment_root / "patch_model").glob("*.ckpt"))[0]
    cp_path = experiment_root / "patch_model" / "checkpoint.ckpt-v0.ckpt"
    classifier = PatchClassifier.load_from_checkpoint(checkpoint_path=cp_path)

    output_dir16 = experiment_root / "inference_results" / "test16"

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    test = SlidesIndex.load(camelyon16.testing(), experiment_root / "test_index")

    transform = Compose([
        RandomCrop((299, 299)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # create list of augmentations 
    augmentations = [Rotate(angle=0), Rotate(angle=90), Rotate(angle=180), Rotate(angle=270),
                     FlipRotate(angle=0), FlipRotate(angle=90), FlipRotate(angle=180), FlipRotate(angle=270)]

    test_results16 = SlidesIndexResults.predict(test, classifier, transform, 128, output_dir16,
                                                 results_dir_name, heatmap_dir_name, augments=augmentations, nthreads=2)
    test_results16.save()


def average_augments_valid() -> None:
    """
    Output from inference contains the values for all 8 augments of each patch
    Need to average them to get the final results for postprocessing
    """
    results_in_name = "results"
    heatmap_in_name = "heatmaps"
    dirin = experiment_root / "inference_results" / 'valid16' 
    dirout = experiment_root / "inference_results" / 'valid16mean' 

    # need a cutdown validation dataset for this experiment
    valid = SlidesIndex.load(camelyon16.training(), experiment_root / "valid_index") 
    cam16_valid = get_subset_of_dataset(valid, camelyon16.training())
    valid16 = SlidesIndexResults.load(cam16_valid, dirin, results_in_name, heatmap_in_name)
    for ps in valid16:
        print(ps.slide_path)
    mean_valid16 = average_patches(valid16, 8, dirout)
    mean_valid16.save(writeps=True)


def average_augments_test() -> None:
    """
    Output from inference contains the values for all 8 augments of each patch
    Need to average them to get the final results for postprocessing
    """
    results_in_name = "results"
    heatmap_in_name = "heatmaps"
    dirin = experiment_root / "inference_results" / 'test16' 
    dirout = experiment_root / "inference_results" / 'test16mean' 

    # need a cutdown validation dataset for this experiment
    test16 = SlidesIndexResults.load(camelyon16.testing(), dirin, results_in_name, heatmap_in_name)
    for ps in test16:
        print(ps.slide_path)
    mean_test16 = average_patches(test16, 8, dirout)
    mean_test16.save(writeps=True)


def calculate_patch_level_results_valid_16() -> None:     

    set_seed(global_seed)

    # need a cutdown validation dataset for this experiment
    valid = SlidesIndex.load(camelyon16.training(), experiment_root / "valid_index") 
    cam16_valid = get_subset_of_dataset(valid, camelyon16.training())

    # define strings for model and split
    results_out_name = "patch_summaries"
    results_in_name = "results"
    heatmap_in_name = "heatmaps"

    # set paths for model and split
    model_dir = experiment_root / "inference_results"
    dirin = model_dir / 'valid16mean'
    dirout = dirin / results_out_name

    # read in predictions
    split_results = SlidesIndexResults.load(cam16_valid, dirin, results_in_name, heatmap_in_name)

    # calculate patch level results
    title16 = experiment_name + ' experiment Camelyon 16 valid dataset'
    patch_level_metrics([split_results], dirout, title16, ci=False)


def calculate_patch_level_results_test_16() -> None:     

    set_seed(global_seed)

    # define strings for model and split
    results_out_name = "patch_summaries"
    results_in_name = "results"
    heatmap_in_name = "heatmaps"

    # set paths for model and split
    model_dir = experiment_root / "inference_results"
    dirin = model_dir / 'test16mean'
    dirout = dirin / results_out_name

    # read in predictions
    split_results = SlidesIndexResults.load(camelyon16.testing(), dirin, results_in_name, heatmap_in_name)

    # calculate patch level results
    title16 = experiment_name + ' experiment Camelyon 16 test dataset'
    patch_level_metrics([split_results], dirout, title16, ci=False)


def calculate_slide_level_results() -> None:
    set_seed(global_seed)

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    validresultsin_post = experiment_root / "inference_results" / "valid16mean" 
    testresultsin_post = experiment_root / "inference_results" / "test16mean" 

    validresults_out_post = experiment_root / "inference_results" / "slide_results16_valid"
    testresults_out_post = experiment_root / "inference_results" / "slide_results16_test"

    title_postv = experiment_name + " experiment, Camelyon 16 valid dataset"
    title_postt = experiment_name + " experiment, Camelyon 16 test dataset"

    #camelyon16_validation = SlidesIndex.load(camelyon16.training(), experiment_root / "valid_index") 
    #camelyon16_validation = get_subset_of_dataset(camelyon16_validation, camelyon16.training())

    #valid_results_post = SlidesIndexResults.load(camelyon16_validation, validresultsin_post, results_dir_name, heatmap_dir_name)
    test_results_post = SlidesIndexResults.load(camelyon16.testing(), testresultsin_post, results_dir_name, heatmap_dir_name)

    slide_classifier = SlideClassifierLiu(camelyon16.training().slide_labels)
    #slide_classifier.calc_features(valid_results_post, validresults_out_post)
    slide_classifier.calc_features(test_results_post, testresults_out_post)
    #slide_classifier.predict_slide_level(features_dir=validresults_out_post)
    slide_classifier.predict_slide_level(features_dir=testresults_out_post)
    #slide_classifier.calc_slide_metrics(title_postv, validresults_out_post)
    slide_classifier.calc_slide_metrics(title_postt, testresults_out_post)
