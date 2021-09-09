import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning import seed_everything
from pytorch_lightning.plugins import DDPPlugin
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
from repath.data.datasets.dataset import Dataset
from repath.preprocess.augmentation.augments import RandomRotateFromList, RandomCropSpecifyOffset
from repath.preprocess.tissue_detection import TissueDetectorOTSU
from repath.preprocess.patching import GridPatchFinder, SlidesIndex, CombinedIndex
from repath.preprocess.sampling import split_camelyon16, balanced_sample, get_subset_of_dataset
from repath.postprocess.results import SlidesIndexResults
from repath.models import GoogLeNet
from repath.utils.seeds import set_seed
from repath.postprocess.patch_level_results import patch_level_metrics
from repath.postprocess.slide_level_metrics import SlideClassifierWang
from repath.postprocess.find_lesions import LesionFinderWang


"""
Global stuff
"""
experiment_name = "wang"
experiment_root = project_root() / "experiments" / experiment_name
tissue_detector = TissueDetectorOTSU()

global_seed = 123

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    set_seed(worker_seed)


class PatchClassifier(pl.LightningModule):
    
    def __init__(self) -> None:
        super().__init__()
        self.model = GoogLeNet(num_classes=2)
        self.model.dropout = nn.Dropout(0.5)

    def training_step(self, batch, batch_idx):
        x, y = batch
        output, aux2, aux1 = self.model(x)
        pred = torch.log_softmax(output, dim=1)

        criterion = nn.CrossEntropyLoss()
        loss1 = criterion(output, y)
        loss2 = criterion(aux1, y)
        loss3 = criterion(aux2, y)
        loss = loss1 + 0.3 * loss2 + 0.3 * loss3
        self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        
        correct=pred.argmax(dim=1).eq(y).sum().item()
        total=len(y)   
        accu = correct / total
        self.log("train_accuracy", accu, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        output = self.model(x)
        pred = torch.log_softmax(output, dim=1)

        criterion = nn.CrossEntropyLoss()
        loss = criterion(output, y)
        self.log("val_loss", loss, on_epoch=True, sync_dist=True)
        
        correct=pred.argmax(dim=1).eq(y).sum().item()
        total=len(y)   
        accu = correct / total
        self.log("val_accuracy", accu, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), 
                                    lr=0.01,
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
    print("read train index")
    valid = SlidesIndex.load(train_data, experiment_root / "valid_index")
    print("read valid index")

    # sample from train and valid sets
    train_samples = balanced_sample([train], 20000)
    print("balanced train sample")
    valid_samples = balanced_sample([valid], 5000)
    print("balanced valid sample")

    train_samples.save(experiment_root / "train_samples2")
    valid_samples.save(experiment_root / "valid_samples2")

    # save out all the patches
    train_samples.save_patches(experiment_root / "training_patches")
    valid_samples.save_patches(experiment_root / "validation_patches")


def train_patch_classifier() -> None:
    """ Trains a classifier on the train patches and validates on validation patches.
    """
    print("Training patch classifier for Wang.")

    set_seed(global_seed)
    #seed_everything(global_seed, workers=True)
    
    # transforms
    transform = Compose([
        RandomRotateFromList([0.0, 90.0, 180.0, 270.0]),
        RandomCropSpecifyOffset(32),
        ToTensor() #,
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # prepare our data
    batch_size = 32
    train_set = ImageFolder(experiment_root / "training_patches", transform=transform)
    valid_set = ImageFolder(experiment_root / "validation_patches", transform=transform)   
    
    # create dataloaders
    g = torch.Generator()
    g.manual_seed(0)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, worker_init_fn=seed_worker, generator=g)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=8, worker_init_fn=seed_worker, generator=g)

    # configure logging and checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy",
        dirpath=experiment_root / "patch_model2",
        filename=f"checkpoint",
        save_last=True,
        mode="max",
    )

    # create a logger
    csv_logger = pl_loggers.CSVLogger(experiment_root / 'logs', name='patch_classifier', version=0)

    # train our model
    #torch.manual_seed(global_seed)
    classifier = PatchClassifier()
    trainer = pl.Trainer(callbacks=[checkpoint_callback], gpus=8, accelerator="ddp", max_epochs=3, 
                     logger=csv_logger, plugins=DDPPlugin(find_unused_parameters=False), deterministic=True)
    trainer.fit(classifier, train_dataloaders=train_loader, val_dataloaders=valid_loader)


def inference_on_train_pre() -> None:
    set_seed(global_seed)
    cp_path = experiment_root / "patch_model" / "checkpoint.ckpt"
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

    patch_finder = GridPatchFinder(labels_level=5, patch_level=0, patch_size=256, stride=256)
    train_patches_grid = SlidesIndex.index_dataset(train_data_cut_down, tissue_detector, patch_finder)
    train_patches_grid.save(experiment_root / "train_index_grid")

    train_patches_grid = SlidesIndex.load(train_data_cut_down, experiment_root / "train_index_grid")

    transform = Compose([
        RandomCropSpecifyOffset(32),
        ToTensor(),
        #Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
        RandomRotateFromList([0.0, 90.0, 180.0, 270.0]),
        RandomCropSpecifyOffset(32),
        ToTensor() #,
        # Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # prepare our data
    batch_size = 32
    train_set = ImageFolder(experiment_root / "training_patches", transform=transform)
    valid_set = ImageFolder(experiment_root / "validation_patches", transform=transform)   
    
    # create dataloaders
    #g = torch.Generator()
    #g.manual_seed(0)
    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, worker_init_fn=seed_worker)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, num_workers=8, worker_init_fn=seed_worker)

    # configure logging and checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy",
        dirpath=experiment_root / "patch_model_hnm",
        filename=f"checkpoint",
        save_last=True,
        mode="max",
    )

    # create a logger
    csv_logger = pl_loggers.CSVLogger(experiment_root / 'logs', name='patch_classifier_hnm', version=0)

    # train our model
    cp_path = experiment_root / "patch_model" / "checkpoint.ckpt"
    #classifier = PatchClassifier.load_from_checkpoint(checkpoint_path=cp_path)
    torch.manual_seed(global_seed)
    classifier = PatchClassifier()
    trainer = pl.Trainer(resume_from_checkpoint=cp_path, callbacks=[checkpoint_callback], gpus=8, accelerator="ddp", max_epochs=3, 
                     logger=csv_logger, deterministic=True)
    trainer.fit(classifier, train_dataloader=train_loader, val_dataloaders=valid_loader)


def inference_on_train_post() -> None:
    set_seed(global_seed)
    cp_path = experiment_root / "patch_model_hnm" / "checkpoint.ckpt"
    classifier = PatchClassifier.load_from_checkpoint(checkpoint_path=cp_path)

    output_dir16 = experiment_root / "post_hnm_results" / "train16"

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    train_overlapping = SlidesIndex.load(camelyon16.training(), experiment_root / "train_index")

    # original training data indexes are overlapping we need non overlapping grid for inference
    train_ol_slides = [pat.slide_path for pat in train_overlapping.patches]
    train_data_cut_down = camelyon16.training()
    mask = [sl in train_ol_slides for sl in train_data_cut_down.paths.slide]
    train_data_cut_down.paths = train_data_cut_down.paths[mask]

    train_patches_grid = SlidesIndex.load(train_data_cut_down, experiment_root / "train_index_grid")

    transform = Compose([
        RandomCropSpecifyOffset(32),
        ToTensor(),
        #Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_results16 = SlidesIndexResults.predict(train_patches_grid, classifier, transform, 128, output_dir16,
                                                 results_dir_name, heatmap_dir_name, nthreads=2)
    train_results16.save()


def inference_on_valid_pre() -> None:
    set_seed(global_seed)
    cp_path = experiment_root / "patch_model" / "checkpoint.ckpt"
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
        RandomCropSpecifyOffset(32),
        ToTensor(),
        #Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    valid_results16 = SlidesIndexResults.predict(valid_patches_grid, classifier, transform, 128, output_dir16,
                                                 results_dir_name, heatmap_dir_name, nthreads=2)
    valid_results16.save()


def inference_on_valid_post() -> None:
    set_seed(global_seed)
    cp_path = experiment_root / "patch_model_hnm" / "checkpoint.ckpt"
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
        RandomCropSpecifyOffset(32),
        ToTensor(),
        #Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
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
    cp_path = experiment_root / "patch_model" / "checkpoint.ckpt"
    classifier = PatchClassifier.load_from_checkpoint(checkpoint_path=cp_path)

    output_dir16 = experiment_root / "pre_hnm_results" / "test16"

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    test_patches = SlidesIndex.load(camelyon16.testing(), experiment_root / "test_index16")
    # using only the grid patches finds only 340 false positives, not enough to retrain so try using overlapping to create more patches

    transform = Compose([
        RandomCropSpecifyOffset(32),
        ToTensor(),
        #Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_results16 = SlidesIndexResults.predict(test_patches, classifier, transform, 128, output_dir16,
                                                 results_dir_name, heatmap_dir_name, nthreads=2)
    test_results16.save()


def inference_on_test_post() -> None:
    set_seed(global_seed)
    cp_path = experiment_root / "patch_model_hnm" / "checkpoint.ckpt"
    classifier = PatchClassifier.load_from_checkpoint(checkpoint_path=cp_path)

    output_dir16 = experiment_root / "post_hnm_results" / "test16"

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    test_patches = SlidesIndex.load(camelyon16.testing(), experiment_root / "test_index16")
    # using only the grid patches finds only 340 false positives, not enough to retrain so try using overlapping to create more patches

    transform = Compose([
        RandomCropSpecifyOffset(32),
        ToTensor(),
        #Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_results16 = SlidesIndexResults.predict(test_patches, classifier, transform, 128, output_dir16,
                                                 results_dir_name, heatmap_dir_name, nthreads=2)
    test_results16.save()


def calculate_patch_level_results() -> None:
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
        patch_level_metrics([split_results_16], splitdirout16, title16, ci=ci)

    set_seed(global_seed)

    valid = SlidesIndex.load(camelyon16.training(), experiment_root / "valid_index") 
    cam16_valid = get_subset_of_dataset(valid, camelyon16.training())

    patch_dataset_function("pre_hnm", "valid", cam16_valid, ci=False)
    patch_dataset_function("pre_hnm", "test", camelyon16.testing(), ci=False)
    patch_dataset_function("post_hnm", "valid", cam16_valid, ci=False)
    patch_dataset_function("post_hnm", "test", camelyon16.testing(), ci=False)


def calculate_slide_level_results_pre() -> None:
    set_seed(global_seed)

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    trainresultsin_pre = experiment_root / "pre_hnm_results" / "train16" 
    validresultsin_pre = experiment_root / "pre_hnm_results" / "valid16" 
    testresultsin_pre = experiment_root / "pre_hnm_results" / "test16" 

    trainresults_out_pre = experiment_root / "pre_hnm_results" / "slide_results16_train"
    validresults_out_pre = experiment_root / "pre_hnm_results" / "slide_results16_valid"
    testresults_out_pre = experiment_root / "pre_hnm_results" / "slide_results16_test"

    title_prev = experiment_name + " experiment, pre hnm model, Camelyon 16 valid dataset"
    title_pret = experiment_name + " experiment, pre hnm model, Camelyon 16 test dataset"

    camelyon16_validation = SlidesIndex.load(camelyon16.training(), experiment_root / "valid_index") 
    camelyon16_validation = get_subset_of_dataset(camelyon16_validation, camelyon16.training())

    train_results_pre = SlidesIndexResults.load(camelyon16.training(), trainresultsin_pre, results_dir_name, heatmap_dir_name)
    valid_results_pre = SlidesIndexResults.load(camelyon16_validation, validresultsin_pre, results_dir_name, heatmap_dir_name)
    test_results_pre = SlidesIndexResults.load(camelyon16.testing(), validresultsin_pre, results_dir_name, heatmap_dir_name)

    slide_classifier = SlideClassifierWang(camelyon16.training().slide_labels)
    slide_classifier.calc_features(train_results_pre, trainresults_out_pre)
    slide_classifier.calc_features(valid_results_pre, validresults_out_pre)
    slide_classifier.calc_features(test_results_pre, testresults_out_pre)
    slide_classifier.predict_slide_level(features_dir=trainresults_out_pre, classifier_dir=trainresults_out_pre, retrain=True)
    slide_classifier.predict_slide_level(features_dir=validresults_out_pre, classifier_dir=trainresults_out_pre, retrain=False)
    slide_classifier.predict_slide_level(features_dir=testresults_out_pre, classifier_dir=trainresults_out_pre, retrain=False)
    slide_classifier.calc_slide_metrics(title_prev, validresults_out_pre)
    slide_classifier.calc_slide_metrics(title_pret, testresults_out_pre)



def calculate_slide_level_results_post() -> None:
    set_seed(global_seed)

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    trainresultsin_post = experiment_root / "post_hnm_results" / "train16" 
    validresultsin_post = experiment_root / "post_hnm_results" / "valid16" 
    testresultsin_post = experiment_root / "post_hnm_results" / "test16" 

    trainresults_out_post = experiment_root / "post_hnm_results" / "slide_results16_train"
    validresults_out_post = experiment_root / "post_hnm_results" / "slide_results16_valid"
    testresults_out_post = experiment_root / "post_hnm_results" / "slide_results16_test"

    title_postv = experiment_name + " experiment, post hnm model, Camelyon 16 valid dataset"
    title_postt = experiment_name + " experiment, post hnm model, Camelyon 16 test dataset"

    camelyon16_validation = SlidesIndex.load(camelyon16.training(), experiment_root / "valid_index") 
    camelyon16_validation = get_subset_of_dataset(camelyon16_validation, camelyon16.training())

    train_results_post = SlidesIndexResults.load(camelyon16.training(), trainresultsin_post, results_dir_name, heatmap_dir_name)
    valid_results_post = SlidesIndexResults.load(camelyon16_validation, validresultsin_post, results_dir_name, heatmap_dir_name)
    test_results_post = SlidesIndexResults.load(camelyon16.testing(), validresultsin_post, results_dir_name, heatmap_dir_name)

    slide_classifier = SlideClassifierWang(camelyon16.training().slide_labels)
    slide_classifier.calc_features(train_results_post, trainresults_out_post)
    slide_classifier.calc_features(valid_results_post, validresults_out_post)
    slide_classifier.calc_features(test_results_post, testresults_out_post)
    slide_classifier.predict_slide_level(features_dir=trainresults_out_post, classifier_dir=trainresults_out_post, retrain=True)
    slide_classifier.predict_slide_level(features_dir=validresults_out_post, classifier_dir=trainresults_out_post, retrain=False)
    slide_classifier.predict_slide_level(features_dir=testresults_out_post, classifier_dir=trainresults_out_post, retrain=False)
    slide_classifier.calc_slide_metrics(title_postv, validresults_out_post)
    slide_classifier.calc_slide_metrics(title_postt, testresults_out_post)


def calculate_lesion_level_results_valid() -> None:
    set_seed(global_seed)

    resultsin_pre = experiment_root / "pre_hnm_results" / "valid16" 
    resultsin_post = experiment_root / "post_hnm_results" / "valid16" 

    results_out_post = experiment_root / "post_hnm_results" / "lesion_results" / "valid16"

    mask_dir = project_root() / 'experiments' / 'masks' / 'camelyon16' / 'training'

    title_post = experiment_name + " experiment, post hnm model, Camelyon 16 valid dataset"

    camelyon16_validation = SlidesIndex.load(camelyon16.training(), experiment_root / "valid_index") 
    camelyon16_validation = get_subset_of_dataset(camelyon16_validation, camelyon16.training())

    valid_results_pre = SlidesIndexResults.load(camelyon16_validation, resultsin_pre, "results", "heatmaps")
    valid_results_post = SlidesIndexResults.load(camelyon16_validation, resultsin_post, "results", "heatmaps")

    # wang example need both pre and post
    lesion_finder = LesionFinderWang(mask_dir, results_out_post)
    lesion_finder.calc_lesions(valid_results_pre, valid_results_post)
    lesion_finder.calc_lesion_results(title_post)
    


def calculate_lesion_level_results_test() -> None:
    set_seed(global_seed)

    resultsin_pre = experiment_root / "pre_hnm_results" / "test16" 
    resultsin_post = experiment_root / "post_hnm_results" / "test16" 

    results_out_post = experiment_root / "post_hnm_results" / "lesion_results" / "test16"

    mask_dir = project_root() / 'experiments' / 'masks' / 'camelyon16' / 'testing'

    title_post = experiment_name + " experiment, post hnm model, Camelyon 16 test dataset"

    test_results_pre = SlidesIndexResults.load(camelyon16.testing(), resultsin_pre, "results", "heatmaps")
    test_results_post = SlidesIndexResults.load(camelyon16.testing(), resultsin_post, "results", "heatmaps")

    # wang example need both pre and post
    lesion_finder = LesionFinderWang(mask_dir, results_out_post)
    lesion_finder.calc_lesions(test_results_pre, test_results_post)
    lesion_finder.calc_lesion_results(title_post)


run_order = [preprocess_indexes, preprocess_samples, train_patch_classifier, inference_on_train_pre, 
create_hnm_patches, retrain_patch_classifier_hnm, inference_on_train_post, inference_on_valid_pre, 
inference_on_valid_post, preprocess_test_index, inference_on_test_pre, inference_on_test_post,
calculate_patch_level_results, calculate_slide_level_results_pre, calculate_slide_level_results_post,
calculate_lesion_level_results_valid, calculate_lesion_level_results_test]
