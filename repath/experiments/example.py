import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import random

from repath.utils.paths import project_root
from repath.utils.convert import remove_item_from_dict
import repath.data.datasets.camelyon16 as camelyon16
import repath.data.datasets.camelyon17 as camelyon17
from repath.preprocess.tissue_detection import TissueDetectorOTSU
from repath.preprocess.patching import GridPatchFinder, SlidesIndex, SlidesIndexResults
from repath.patch_classification.models.simple import Backbone
from repath.postprocess.slide_dataset import SlideDataset
from repath.postprocess.prediction import inference_on_slide
from repath.preprocess.sampling import split_camelyon16, split_camelyon17, balanced_sample
from repath.postproces.patch_level_results import patch_level_metrics

from repath.utils.seeds import set_seed

"""
Global stuff
"""
experiment_name = "example"
experiment_root = project_root() / "experiments" / experiment_name
tissue_detector = TissueDetectorOTSU()


global_seed = 123


class PatchClassifier(pl.LightningModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def accuracy(self, logits, labels):
        _, pred = torch.max(logits, 1)
        accuracy = Accuracy()
        accu = accuracy(pred, labels)
        return accu

    def step(self, batch, batch_idx, label):
        x, y = batch
        logits = self.model(x)
        x = torch.log_softmax(x, dim=1)
        loss = self.cross_entropy_loss(logits, y)
        accu = self.accuracy(logits, y)
        self.log(f"{label}_loss", loss)
        self.log(f"{label}_accuracy", accu)
        return loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self.step(batch, batch_idx, "val")

    def configure_optimizers(self):
        optimizer = torch.opt
        return optimizer


"""
Sections of the experiment
"""


def preprocesses() -> None:
    """Performs preprocess steps:
        -Getting train slides and their associated annotation files from folders. 
        -Index to all the patches in the dataset
        -split patches to train and validation sets
        -saving the patches in respective folder
    """
    set_seed(global_seed)
    # index all the patches for the camelyon16 dataset
    train_data = camelyon16.training()
    patch_finder = GridPatchFinder(6, 0, 256, 256)
    train_patches = SlidesIndex.index_dataset(train_data, tissue_detector, patch_finder)

    # do the train validate split
    train, valid = split_camelyon16(train_patches, 0.7)
    train.save(experiment_root / "train_index")
    valid.save(experiment_root / "valid_index")

    # sample from train and valid sets
    train_samples = balanced_sample(train, 700000)
    valid_samples = balanced_sample(valid, 300000)

    # save out all the patches
    train_samples.save_patches(experiment_root / "training_patches")
    valid_samples.save_patches(experiment_root / "validation_patches")


def train_patch_classifier() -> None:
    """Trains a patch classifier on train patches and validates on validation patches.
    """
    set_seed(global_seed)
    # prepare our data
    batch_size = 128
    train_set = ImageFolder.load(experiment_root / "training_patches")
    valid_set = ImageFolder.load(experiment_root / "validation_patches")
    train_loader = DataLoader(train_set, batch_size=batch_size, worker_init_fn=np.random.seed(global_seed))
    valid_loader = DataLoader(valid_set, batch_size=batch_size, worker_init_fn=np.random.seed(global_seed))

    # configure logging and checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=experiment_root / "patch_model",
        filename=f"checkpoint-{epoch:02d}-{val_loss:.2f}.ckpt",
        save_top_k=1,
        mode="min",
    )

    # train our model
    model = Backbone()
    classifier = PatchClassifier(model)
    trainer = pl.Trainer(callbacks=[checkpoint_callback])
    trainer.fit(classifier, train_dataloader=train_loader, val_dataloaders=valid_loader)


def inference_on_train() -> None:
    """Performs inference on train patches using the initial trained mode.
    """
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

    train_results16 = SlidesIndexResults.predict_dataset(train16, classifier, 128, 8, transform, output_dir16, results_dir_name, heatmap_dir_name)
    train_results16.save_results_index()

    train_results17 = SlidesIndexResults.predict_dataset(train17, classifier, 128, 8, transform, output_dir17, results_dir_name, heatmap_dir_name)
    train_results17.save_results_index()


def create_hnm_patches() -> None:
    """Creates extra patches doing hard negative mining.
    """
    set_seed(global_seed)

    input_dir16 = experiment_root / "train_index16" / "pre_hnm_results"
    input_dir17 = experiment_root / "train_index17" / "pre_hnm_results"

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    train_results16 = SlidesIndexResults.load_results_index(camelyon16.training(), input_dir16, results_dir_name, heatmap_dir_name)
    train_results17 = SlidesIndexResults.load_results_index(camelyon17.training(), input_dir17, results_dir_name, heatmap_dir_name)

    train_results = CombinedIndex.for_slide_indexes([train_results16, train_results17])

    FP_mask = np.logical_and(train_results.patches_df['tumor'] > 0.5, train_results.patches_df['label'] == 1)

    hnm_patches_df = train_results.patches_df[FP_mask]
    hnm_patches_df = hnm_patches_df.sort_values('tumor', axis=0, ascending=False)
    ### limit number of patches to same number as original patches
    hnm_patches_df = hnm_patches_df.iloc[0:47574]

    train_results.patches_df = hnm_patches_df

    train_results.save_patches(experiment_root / "training_patches", affix='-hnm')


def preprocess_for_testing() -> None:
    """Performa all preprocess steps on testing datasets which includes:
        - Getting test slides from respective folders.
        -Index to all the patches in the dataset
        -saving the patches in respective folder
    """
    set_seed(global_seed)

    # index all the patches for the camelyon16 dataset
    test_data = camelyon16.testing()
    patch_finder = GridPatchFinder(6, 0, 256, 256)
    test_patches = SlidesIndex.index_dataset(test_data, tissue_detector, patch_finder)
    test_patches.save(experiment_root / "test_index")

    # index all the patches for the camelyon17 dataset
    train_data17 = camelyon17.training()
    test_data17 = camelyon17.testing()
    patch_finder = GridPatchFinder(6, 0, 256, 256)
    train_patches17 = SlidesIndex.index_dataset(train_data17, tissue_detector, patch_finder)
    test_patches17 = SlidesIndex.index_dataset(test_data17, tissue_detector, patch_finder)
    test_patches17.save(experiment_root / "test_index_17")

    # do the train validate split
    train17, valid17 = split_camelyon17(train_patches17, 0.7)

    # save the train and valid patch indexes
    valid17.save(experiment_root / "valid_index_17")


def inference_on_slides() -> None:
    set_seed(global_seed)

    batsz = 128
    nwork = 80
    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    # Camelyon 16 details
    valid16_dir_in = experiment_root / "valid_index"
    valid16 = SlidesIndex.load(camelyon16.training(), valid16_dir_in)
    test16_dir_in = experiment_root / "test_index"
    test16 = SlidesIndex.load(camelyon16.testing(), test16_dir_in)

    # Camelyon 17 details
    valid17_dir_in = experiment_root / "valid_index_17"
    valid17 = SlidesIndex.load(camelyon17.training(), valid17_dir_in)
    test17_dir_in = experiment_root / "test_index_17"
    test17 = SlidesIndex.load(camelyon17.testing(), test17_dir_in)

    # Pre hnm details
    cp_path_pre_hnm = (experiment_root / "patch_model").glob("*.ckpt")[0]
    classifier_pre_hnm = PatchClassifier.load_from_checkpoint(
        checkpoint_path=cp_path_pre_hnm, model=Backbone()
    )
    outdir_val16_pre = experiment_root / "pre_hnm_results" / "valid16"
    outdir_val17_pre = experiment_root / "pre_hnm_results" / "valid17"
    outdir_tst16_pre = experiment_root / "pre_hnm_results" / "test16"
    outdir_tst17_pre = experiment_root / "pre_hnm_results" / "test17"

    # Post hnm details
    cp_path_post_hnm = (experiment_root / "patch_model_hnm").glob("*.ckpt")[0]
    classifier_post_hnm = PatchClassifier.load_from_checkpoint(
        checkpoint_path=cp_path_post_hnm, model=Backbone()
    )
    outdir_val16_post = experiment_root / "post_hnm_results" / "valid16"
    outdir_val17_post = experiment_root / "post_hnm_results" / "valid17"
    outdir_tst16_post = experiment_root / "post_hnm_results" / "test16"
    outdir_tst17_post = experiment_root / "post_hnm_results" / "test17"

    # pre hnm results
    valid_results_16_pre = SlidesIndexResults.predict_dataset(valid16, classifier_pre_hnm, batsz, nwork,
                                                              outdir_val16_pre, results_dir_name, heatmap_dir_name)
    valid_results_16_pre.save_results_index()
    valid_results_17_pre = SlidesIndexResults.predict_dataset(valid17, classifier_pre_hnm, batsz, nwork,
                                                              outdir_val17_pre, results_dir_name, heatmap_dir_name)
    valid_results_17_pre.save_results_index()
    test_results_16_pre = SlidesIndexResults.predict_dataset(test16, classifier_pre_hnm, batsz, nwork,
                                                             outdir_tst16_pre, results_dir_name, heatmap_dir_name)
    test_results_16_pre.save_results_index()
    test_results_17_pre = SlidesIndexResults.predict_dataset(test16, classifier_pre_hnm, batsz, nwork,
                                                             outdir_tst17_pre, results_dir_name, heatmap_dir_name)
    test_results_17_pre.save_results_index()

    # post hnm results
    valid_results_16_post = SlidesIndexResults.predict_dataset(valid16, classifier_post_hnm, batsz, nwork,
                                                              outdir_val16_post, results_dir_name, heatmap_dir_name)
    valid_results_16_post.save_results_index()
    valid_results_17_post = SlidesIndexResults.predict_dataset(valid17, classifier_post_hnm, batsz, nwork,
                                                              outdir_val17_post, results_dir_name, heatmap_dir_name)
    valid_results_17_post.save_results_index()
    test_results_16_post = SlidesIndexResults.predict_dataset(test16, classifier_post_hnm, batsz, nwork,
                                                              outdir_tst16_post, results_dir_name, heatmap_dir_name)
    test_results_16_post.save_results_index()
    test_results_17_post = SlidesIndexResults.predict_dataset(test17, classifier_post_hnm, batsz, nwork,
                                                              outdir_tst17_post, results_dir_name, heatmap_dir_name)
    test_results_17_post.save_results_index()


def calculate_patch_level_results() -> None:
    def patch_dataset_function(modelname: str, splitname: str, dataset16: SlideDataset, dataset17: SlideDataset):
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
        title16 = experiment_name + ' experiment ' + modelname + ' model Camelyon 16 ' + splitname ' dataset'
        patch_level_metrics([split_results_16], splitdirout16, title16, ci=False)
        title17 = experiment_name + ' experiment ' + modelname + ' model Camelyon 17 ' + splitname ' dataset'
        patch_level_metrics([split_results_17], splitdirout17, title17, ci=False)
        title1617 = experiment_name + ' experiment ' + modelname + ' model Camelyon 16 & 17 ' + splitname ' dataset'
        patch_level_metrics([split_results_1617], splitdirout1617, title1617, ci=False)

    set_seed(global_seed)

    patch_dataset_function("pre_hnm", "valid", camelyon16.training(), camelyon17.training(), ci=True)
    patch_dataset_function("pre_hnm", "test", camelyon16.testing(), camelyon17.testing(), ci=True)
    patch_dataset_function("post_hnm", "valid", camelyon16.training(), camelyon17.training(), ci=True)
    patch_dataset_function("post_hnm", "test", camelyon16.testing(), camelyon17.testing(), ci=True)


def calculate_lesion_level_results() -> None:
    set_seed(global_seed)

    results_pre_hnm = experiment_root / "valid_index" / "pre_hnm_results"
    results_post_hnm = experiment_root / "valid_index" / "post_hnm_results"

    valid_results_pre = SlidesIndexResults.load(camelyon16.training(), results_pre_hnm)
    valid_results_post = SlidesIndexResults.load(camelyon16.training(), results_pre_hnm)

    lesions_all_slides = pd.DataFrame(columns=["prob_score", "centre_row", "centre_col", "pixels", "filename"])

    for result_pre, result_post in zip(valid_results_pre, valid_results_post):
        heatmap_pre = result_pre.to_heatmap
        heatmap_post = result_post.to_heatmap
        lesion_labelled_image = ConnectedComponents(0.9).segment(heatmap_pre)
        lesions_out = LesionFinderWang().find_lesions(heatmap_pre, heatmap_post, lesion_labelled_image)
        lesions_out["filename"] = result_pre.slide_path.stem

        lesions_all_slides = pd.concat((lesions_all_slides, lesions_out), axis=0, ignore_index=True)

    froc_curve, froc = evaluate_froc(paper.mask_dir, lesions_all_slides, 5, 0.243)
    froc_curve.to_csv(results_dir / 'froc_curve.csv', index=False)
    froc_plot = plotROC(froc_curve.total_FPs, froc_curve.total_sensitivity, froc,
                        "Free Receiver Operating Characteristic Curve for Lesion Detection",
                        "Average False Positives", "Metastatis Detection Sensitivity", [0, 8])
    froc_plot.savefig(results_dir / 'froc_curve.png')

# using the slide sampling validation set, predict a probability for every patch to create a heat map

# there are patch results (can use slides and patches sets), slide results, lesion level results, patient level results

steps = [preprocessing, patch_training]
