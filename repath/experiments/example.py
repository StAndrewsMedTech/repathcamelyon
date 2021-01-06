import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch

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


"""
Global stuff
"""
experiment_name = "example"
experiment_root = project_root() / "experiments" / experiment_name
tissue_detector = TissueDetectorOTSU()


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
    # prepare our data
    batch_size = 128
    train_set = ImageFolder.load(experiment_root / "training_patches")
    valid_set = ImageFolder.load(experiment_root / "validation_patches")
    train_loader = DataLoader(train_set, batch_size=batch_size)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)

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


def hard_negative_mining_and_retrain() -> None:

    # calculate false positives
    cp_path = list((experiment_root / "patch_model").glob("*.ckpt"))[0]
    classifier = PatchClassifier.load_from_checkpoint(
        checkpoint_path=cp_path, model=Backbone()
    )

    output_dir = experiment_root / "train_index"
    results_dir_name = "pre_hnm_results"
    heatmap_dir_name = "pre_hnm_heatmaps"

    transform = Compose([
        RandomCrop((240, 240)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train = SlidesIndex.load(camelyon16.training(), experiment_root / "train_index")
    train_results = SlidesIndexResults.predict_dataset(train, classifier, 128, 80, transform, output_dir, results_dir_name,
                                                       heatmap_dir_name)
    train_results.save_results_index()

    train_results = train_results.as_combined()

    patch_dict = train_results.dataset.labels
    FP_mask = np.logical_and(train_results.patches_df['tumor'] > 0.5,
                             train_results.patches_df['label'] == patch_dict['tumor'])
    train_results.patches_df = train_results.patches_df[FP_mask]
    ### To do put in a limit on how many patches to add
    train_results.save_patches(experiment_root / "training_patches")

    # prepare our data
    batch_size = 128
    train_set = ImageFolder.load(experiment_root / "training_patches")
    valid_set = ImageFolder.load(experiment_root / "validation_patches")
    train_loader = DataLoader(train_set, batch_size=batch_size)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)

    # configure logging and checkpoints
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=experiment_root / "patch_model_hnm",
        filename=f"checkpoint-{epoch:02d}-{val_loss:.2f}.ckpt",
        save_top_k=1,
        mode="min",
    )

    # train our model
    cp_path = (experiment_root / "patch_model").glob("*.ckpt")[0]
    model = Backbone().load_from_checkpoint(cp_path)
    classifier = PatchClassifier(model)
    trainer = pl.Trainer(callbacks=[checkpoint_callback])
    trainer.fit(classifier, train_dataloader=train_loader, val_dataloaders=valid_loader)


def preprocess_for_testing() -> None:
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

    batsz = 128
    nwork = 80
    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    # Camelyon 16 details
    outdir_val16 = experiment_root / "valid_index"
    valid16 = SlidesIndex.load(camelyon16.training(), outdir_val16)
    outdir_tst16 = experiment_root / "test_index"
    test16 = SlidesIndex.load(camelyon16.testing(), outdir_tst16)

    # Camelyon 17 details
    outdir_val17 = experiment_root / "valid_index_17"
    valid17 = SlidesIndex.load(camelyon17.training(), outdir_val17)
    outdir_tst17 = experiment_root / "test_index_17"
    test17 = SlidesIndex.load(camelyon17.testing(), outdir_tst17)

    # Pre hnm details
    cp_path_pre_hnm = (experiment_root / "patch_model").glob("*.ckpt")[0]
    classifier_pre_hnm = PatchClassifier.load_from_checkpoint(
        checkpoint_path=cp_path_pre_hnm, model=Backbone()
    )
    outdir_val16_pre = outdir_val16 / "pre_hnm_results"
    outdir_val17_pre = outdir_val17 / "pre_hnm_results"
    outdir_tst16_pre = outdir_tst16 / "pre_hnm_results"
    outdir_tst17_pre = outdir_tst17 / "pre_hnm_results"

    # Post hnm details
    cp_path_post_hnm = (experiment_root / "patch_model").glob("*.ckpt")[0]
    classifier_post_hnm = PatchClassifier.load_from_checkpoint(
        checkpoint_path=cp_path_post_hnm, model=Backbone()
    )
    outdir_val16_post = outdir_val16 / "post_hnm_results"
    outdir_val17_post = outdir_val17 / "post_hnm_results"
    outdir_tst16_post = outdir_tst16 / "post_hnm_results"
    outdir_tst17_post = outdir_tst17 / "post_hnm_results"

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

    results_dir_name = "results"
    heatmap_dir_name = "heatmaps"

    # Camelyon 16 details
    outdir_val16 = experiment_root / "valid_index"
    outdir_tst16 = experiment_root / "test_index"
    # Camelyon 17 details
    outdir_val17 = experiment_root / "valid_index_17"
    outdir_tst17 = experiment_root / "test_index_17"

    # Pre hnm details
    outdir_val16_pre = outdir_val16 / "pre_hnm_results"
    outdir_val17_pre = outdir_val17 / "pre_hnm_results"
    outdir_tst16_pre = outdir_tst16 / "pre_hnm_results"
    outdir_tst17_pre = outdir_tst17 / "pre_hnm_results"

    # Post hnm details
    outdir_val16_post = outdir_val16 / "post_hnm_results"
    outdir_val17_post = outdir_val17 / "post_hnm_results"
    outdir_tst16_post = outdir_tst16 / "post_hnm_results"
    outdir_tst17_post = outdir_tst17 / "post_hnm_results"


    valid_results_16_pre = SlidesIndexResults.load_results_index(camelyon16.training(), outdir_val16_pre,
                                                                 results_dir_name, heatmap_dir_name)
    valid_results_16_pre = valid_results_16_pre.as_combined()



def calculate_lesion_level_results() -> None:

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
