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
from repath.preprocess.augmentation.augments import RandomRotateFromList, RandomCropSpecifyOffset
from repath.preprocess.tissue_detection import TissueDetectorGreyScale
from repath.preprocess.patching import GridPatchFinder, SlidesIndex
from repath.preprocess.sampling import split_camelyon16, split_camelyon17, balanced_sample, get_subset_of_dataset
from repath.postprocess.results import SlidesIndexResults
from repath.postprocess.patch_level_results import patch_level_metrics
from repath.postprocess.slide_level_metrics import SlideClassifierLee
from repath.postprocess.patient_level_metrics import calc_patient_level_metrics
from repath.utils.seeds import set_seed, seed_worker
from repath.postprocess.find_lesions import LesionFinderLee

"""
Global stuff
"""
experiment_name = "lee"
experiment_root = project_root() / "experiments" / experiment_name
tissue_detector = TissueDetectorGreyScale()

global_seed = 123

class PatchClassifier(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        model = densenet121(pretrained=True)
        model.classifier = nn.Sequential(
            nn.Linear(in_features=1024, out_features=1000, bias=True),
            nn.Linear(1000, 2))
        self.model = model

    def step(self, batch, batch_idx, label):
        x, y = batch
        logits = self.model(x)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, y)
        self.log(f"{label}_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        pred = torch.log_softmax(logits, dim=1)
        correct = pred.argmax(dim=1).eq(y).sum().item()
        total = len(y)
        accu = correct / total
        self.log(f"{label}_accuracy", loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(),
                                    lr=0.1,
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
        RandomCropSpecifyOffset(16),
        ToTensor() #,
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # prepare our data
    batch_size = 64
    train_set = ImageFolder(experiment_root / "training_patches", transform=transform)
    valid_set = ImageFolder(experiment_root / "validation_patches", transform=transform)

    g = torch.Generator()
    g.manual_seed(0)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=8, worker_init_fn=seed_worker, generator=g)

    # configure logging and checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy",
        dirpath=experiment_root / "patch_model",
        filename=f"checkpoint",
        save_last=True,
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
    trainer = pl.Trainer(callbacks=[checkpoint_callback], gpus=8, accelerator="ddp", max_epochs=15,
                         logger=csv_logger, plugins=DDPPlugin(find_unused_parameters=False), deterministic=True)
    trainer.fit(classifier, train_dataloaders=train_loader, val_dataloaders=valid_loader)


def inference_on_train() -> None:
    set_seed(global_seed)
    cp_path = cp_path = experiment_root / "patch_model" / "checkpoint.ckpt"
    classifier = PatchClassifier.load_from_checkpoint(checkpoint_path=cp_path)

    output_dir16 = experiment_root / "train_index16" / "pre_hnm_results"
    output_dir17 = experiment_root / "train_index17" / "pre_hnm_results"

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    train16 = SlidesIndex.load(camelyon16.training(), experiment_root / "train_index16")
    train17 = SlidesIndex.load(camelyon17.training(), experiment_root / "train_index17")

    transform = Compose([
        RandomCropSpecifyOffset(16),
        ToTensor(),
        #Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_results16 = SlidesIndexResults.predict(train16, classifier, transform, 128, output_dir16,
                                                         results_dir_name, heatmap_dir_name, nthreads=2, 
                                                         global_seed=global_seed)
    train_results16.save()

    train_results17 = SlidesIndexResults.predict(train17, classifier, transform, 128, output_dir17,
                                                         results_dir_name, heatmap_dir_name, nthreads=2, 
                                                         global_seed=global_seed)
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

    train_results.save_patches(experiment_root / "training_patches", affix='-hnm', add_patches=True)


def retrain_patch_classifier_hnm() -> None:
    set_seed(global_seed)
    # transforms
    transform = Compose([
        RandomCropSpecifyOffset(16),
        ToTensor() #,
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # prepare our data
    batch_size = 64
    train_set = ImageFolder(experiment_root / "training_patches", transform=transform)
    valid_set = ImageFolder(experiment_root / "validation_patches", transform=transform)
    g = torch.Generator()
    g.manual_seed(0)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=8, worker_init_fn=seed_worker, generator=g)

    # configure logging and checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy",
        dirpath=experiment_root / "patch_model_hnm",
        filename=f"hnm_checkpoint",
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
    cp_path = cp_path = experiment_root / "patch_model" / "checkpoint.ckpt"
    optimizer = torch.optim.SGD(classifier.parameters(),
                            lr=0.01,
                            momentum=0.9,
                            weight_decay=0.0001)
    classifier = PatchClassifier()
    trainer = pl.Trainer(resume_from_checkpoint=cp_path, callbacks=[checkpoint_callback], gpus=8, accelerator="ddp", max_epochs=15,
                         logger=csv_logger, deterministic=True)
    trainer.fit(classifier, train_dataloader=train_loader, val_dataloaders=valid_loader)
    

def inference_on_train() -> None:
    set_seed(global_seed)
    cp_path = experiment_root / "patch_model_hnm" / "hnm_checkpoint.ckpt"
    classifier = PatchClassifier.load_from_checkpoint(checkpoint_path=cp_path)

    output_dir16 = experiment_root / "post_hnm_results" / "train16"
    output_dir17 = experiment_root / "post_hnm_results" / "train17"

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    train16 = SlidesIndex.load(camelyon16.training(), experiment_root / "train_index16")
    train17 = SlidesIndex.load(camelyon17.training(), experiment_root / "train_index17")

    # valid16.patches = valid16[0:32]
    # valid17.patches = valid17[0:32]

    transform = Compose([
        RandomCropSpecifyOffset(16),
        ToTensor(),
        #Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_results16 = SlidesIndexResults.predict(train16, classifier, transform, 128, output_dir16,
                                                         results_dir_name, heatmap_dir_name, nthreads=2, global_seed=global_seed)
    train_results16.save()

    train_results17 = SlidesIndexResults.predict(train17, classifier, transform, 128, output_dir17,
                                                         results_dir_name, heatmap_dir_name, nthreads=2, global_seed=global_seed)
    train_results17.save()


def inference_on_valid() -> None:
    set_seed(global_seed)
    cp_path = experiment_root / "patch_model_hnm" / "hnm_checkpoint.ckpt"
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
        RandomCropSpecifyOffset(16),
        ToTensor(),
        #Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    valid_results16 = SlidesIndexResults.predict(valid16, classifier, transform, 128, output_dir16,
                                                         results_dir_name, heatmap_dir_name, nthreads=2, global_seed=global_seed)
    valid_results16.save()

    valid_results17 = SlidesIndexResults.predict(valid17, classifier, transform, 128, output_dir17,
                                                         results_dir_name, heatmap_dir_name, nthreads=2, global_seed=global_seed)
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
    cp_path = experiment_root / "patch_model_hnm" / "hnm_checkpoint.ckpt"
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
        RandomCropSpecifyOffset(16),
        ToTensor(),
        #Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_results16 = SlidesIndexResults.predict(test16, classifier, transform, 128, output_dir16,
                                                         results_dir_name, heatmap_dir_name, nthreads=2, global_seed=global_seed)
    test_results16.save()

    test_results17 = SlidesIndexResults.predict(test17, classifier, transform, 128, output_dir17,
                                                         results_dir_name, heatmap_dir_name, nthreads=2, global_seed=global_seed)
    test_results17.save()


def calculate_patch_level_results() -> None:
    def patch_dataset_function(modelname: str, splitname: str, dataset16: Dataset, dataset17: Dataset, ci: bool = False):
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
        split_results_16 = SlidesIndexResults.load(dataset16, splitdirin16,
                                                                 results_in_name, heatmap_in_name)

        split_results_17 = SlidesIndexResults.load(dataset17, splitdirin17,
                                                                 results_in_name, heatmap_in_name)
        split_results_17_annotated_list = [ps for ps in split_results_17 if 'annotated' in ps.tags]
        split_results_17_annotated = split_results_17
        split_results_17_annotated.patches = split_results_17_annotated_list

        # calculate patch level results
        title16 = experiment_name + ' experiment ' + modelname + ' model Camelyon 16 ' + splitname + ' dataset'
        patch_level_metrics([split_results_16], splitdirout16, title16, ci=ci)
        print(len(split_results_17_annotated))
        if len(split_results_17_annotated) > 0:
            title17 = experiment_name + ' experiment ' + modelname + ' model Camelyon 17 ' + splitname + ' dataset'
            patch_level_metrics([split_results_17_annotated], splitdirout17, title17, ci=ci)
            title1617 = experiment_name + ' experiment ' + modelname + ' model Camelyon 16 & 17 ' + splitname + ' dataset'
            patch_level_metrics([split_results_16, split_results_17_annotated], splitdirout1617, title1617, ci=ci)

    set_seed(global_seed)

    #patch_dataset_function("pre_hnm", "valid", camelyon16.training(), camelyon17.training(), ci=False)
    #patch_dataset_function("pre_hnm", "test", camelyon16.testing(), camelyon17.testing(), ci=False)
    patch_dataset_function("post_hnm", "valid", camelyon16.training(), camelyon17.training(), ci=False)
    patch_dataset_function("post_hnm", "test", camelyon16.testing(), camelyon17.testing(), ci=False)


def calculate_slide_level_results() -> None:
    set_seed(global_seed)

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    valid = SlidesIndex.load(camelyon16.training(), experiment_root / "valid_index16") 

    resultsin_post16_train = experiment_root / "post_hnm_results" / "train16"
    resultsin_post17_train = experiment_root / "post_hnm_results" / "train17"
    resultsin_post16_valid = experiment_root / "post_hnm_results" / "valid16"
    resultsin_post17_valid = experiment_root / "post_hnm_results" / "valid17"
    resultsin_post16_test = experiment_root / "post_hnm_results" / "test16"
    resultsin_post17_test = experiment_root / "post_hnm_results" / "test17"

    results_out_post16tr = experiment_root / "post_hnm_results" / "slide_results16_train"
    results_out_post17tr = experiment_root / "post_hnm_results" / "slide_results17_train"
    results_out_post16v = experiment_root / "post_hnm_results" / "slide_results16_valid"
    results_out_post17v = experiment_root / "post_hnm_results" / "slide_results17_valid"
    results_out_post16t = experiment_root / "post_hnm_results" / "slide_results16_test"
    results_out_post17t = experiment_root / "post_hnm_results" / "slide_results17_test"

    title_post16v = experiment_name + " experiment, post hnm model, Camelyon 16 valid dataset"
    title_post17v = experiment_name + " experiment, post hnm model, Camelyon 17 valid dataset"
    title_post16t = experiment_name + " experiment, post hnm model, Camelyon 16 test dataset"
    title_post17t = experiment_name + " experiment, post hnm model, Camelyon 17 test dataset"

    #valid_results_post16tr = SlidesIndexResults.load(camelyon16.training(), resultsin_post16_train, results_dir_name, heatmap_dir_name)
    #valid_results_post17tr = SlidesIndexResults.load(camelyon17.training(), resultsin_post17_train, results_dir_name, heatmap_dir_name)
    #valid_results_post16v = SlidesIndexResults.load(camelyon16.training(), resultsin_post16_valid, results_dir_name, heatmap_dir_name)
    #valid_results_post17v = SlidesIndexResults.load(camelyon17.training(), resultsin_post17_valid, results_dir_name, heatmap_dir_name)
    #valid_results_post16t = SlidesIndexResults.load(camelyon16.testing(), resultsin_post16_test, results_dir_name, heatmap_dir_name)
    #valid_results_post17t = SlidesIndexResults.load(camelyon17.testing(), resultsin_post17_test, results_dir_name, heatmap_dir_name)

    slide_classifier_16 = SlideClassifierLee(camelyon16.training().slide_labels)
    #slide_classifier_16.calc_features(valid_results_post16tr, results_out_post16tr)
    #slide_classifier_16.calc_features(valid_results_post16v, results_out_post16v)
    #slide_classifier_16.calc_features(valid_results_post16t, results_out_post16t)
    #slide_classifier_16.predict_slide_level(features_dir=results_out_post16tr, classifier_dir=results_out_post16tr, retrain=True)
    #slide_classifier_16.predict_slide_level(features_dir=results_out_post16v, classifier_dir=results_out_post16tr, retrain=False)
    #slide_classifier_16.predict_slide_level(features_dir=results_out_post16t, classifier_dir=results_out_post16tr, retrain=False)
    slide_classifier_16.calc_slide_metrics(title_post16v, results_out_post16v)
    slide_classifier_16.calc_slide_metrics(title_post16t, results_out_post16t)

    slide_classifier_17 = SlideClassifierLee(camelyon17.training().slide_labels)
    #slide_classifier_17.calc_features(valid_results_post17tr, results_out_post17tr)
    #slide_classifier_17.calc_features(valid_results_post17v, results_out_post17v)
    #slide_classifier_17.calc_features(valid_results_post17t, results_out_post17t)
    #slide_classifier_17.predict_slide_level(features_dir=results_out_post17tr, classifier_dir=results_out_post17tr, retrain=True)
    #slide_classifier_17.predict_slide_level(features_dir=results_out_post17v, classifier_dir=results_out_post17tr, retrain=False)
    #slide_classifier_17.predict_slide_level(features_dir=results_out_post17t, classifier_dir=results_out_post17tr, retrain=False)
    slide_classifier_17.calc_slide_metrics(title_post17v, results_out_post17v, labelorder=['negative','itc', 'micro', 'macro'])
    slide_classifier_17.calc_slide_metrics(title_post17t, results_out_post17t, labelorder=['negative','itc', 'micro', 'macro'])


def calculate_patient_level_metrics() -> None:
    slide_results_post17v = experiment_root / "post_hnm_results" / "slide_results17_valid"
    slide_results_post17t = experiment_root / "post_hnm_results" / "slide_results17_test"

    patient_results_post17v = experiment_root / "post_hnm_results" / "patient_results17_valid"
    patient_results_post17t = experiment_root / "post_hnm_results" / "patient_results17_test"

    title_post17v = experiment_name + " experiment, post hnm model, Camelyon 17 valid dataset"
    title_post17t = experiment_name + " experiment, post hnm model, Camelyon 17 test dataset"

    calc_patient_level_metrics(input_dir=slide_results_post17v, output_dir=patient_results_post17v, title=title_post17v, ci=False)
    calc_patient_level_metrics(input_dir=slide_results_post17t, output_dir=patient_results_post17t, title=title_post17t, ci=False)


def calculate_lesion_level_results() -> None:
    set_seed(global_seed)

    resultsin_post_v = experiment_root / "post_hnm_results" / "valid16"
    resultsin_post_t = experiment_root / "post_hnm_results" / "test16"

    results_out_post_v = experiment_root / "post_hnm_results" / "lesion_results" / "valid16"
    results_out_post_t = experiment_root / "post_hnm_results" / "lesion_results" / "test16"

    mask_dir_v = project_root() / 'experiments' / 'masks' / 'camelyon16' / 'training'
    mask_dir_t = project_root() / 'experiments' / 'masks' / 'camelyon16' / 'testing'

    title_post_v = experiment_name + " experiment, post hnm model, Camelyon 16 valid dataset"
    title_post_t = experiment_name + " experiment, post hnm model, Camelyon 16 test dataset"

    #camelyon16_validation = SlidesIndex.load(camelyon16.training(), experiment_root / "valid_index16") 
    #camelyon16_validation = get_subset_of_dataset(camelyon16_validation, camelyon16.training())

    valid_results_post = SlidesIndexResults.load(camelyon16.training(), resultsin_post_v, "results", "heatmaps")
    test_results_post = SlidesIndexResults.load(camelyon16.testing(), resultsin_post_t, "results", "heatmaps")

    lesion_finder_v_post = LesionFinderLee(mask_dir_v, results_out_post_v)
    #lesion_finder_v_post.calc_lesions(valid_results_post)
    lesion_finder_v_post.calc_lesion_results(title_post_v)
    lesion_finder_t_post = LesionFinderLee(mask_dir_t, results_out_post_t)
    #lesion_finder_t_post.calc_lesions(test_results_post)
    lesion_finder_t_post.calc_lesion_results(title_post_t)

