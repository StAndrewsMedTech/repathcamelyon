import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.metrics import Accuracy
import torch

from repath.utils.paths import project_root
from repath.utils.convert import remove_item_from_dict
from repath.preprocess.patching import PatchIndex
import repath.data.datasets.camelyon16 as camelyon16
from repath.preprocess.tissue_detection import TissueDetectorOTSU
from repath.preprocess.patching import GridPatchFinder, PatchSet, PatchIndex, SlidePatchSet
from repath.patch_classification.models.simple import Backbone
from repath.postprocess.slide_dataset import SlideDataset
from repath.postprocess.prediction import inference_on_slide
from repath.postprocess.patch_index_results import PatchSetResults, PatchIndexResults
from repath.preprocess.sampling import split_camelyon16, balanced_sample


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
    train_patches = PatchIndex.for_dataset(train_data, tissue_detector, patch_finder)

    # do the train validate split
    train, valid = split_camelyon16(train_patches, 0.7, seperate_slides=True)
    train_samples = balanced_sample(train, 700000)
    valid_samples = balanced_sample(valid, 300000)

    # save the train and valid patch indexes
    train.save(experiment_root / "train_index")
    valid.save(experiment_root / "valid_index")

    # save out the patch sets
    train_samples.save(experiment_root / "training_patches")
    valid_samples.save(experiment_root / "validation_patches")

    # save out all the patches
    train_samples.save_patches(experiment_root / "training_patches")
    valid_samples.save_patches(experiment_root / "validation_patches")


def train_patch_classifier() -> None:
    # prepare our data
    batch_size = 128
    train_set = PatchSet.load(experiment_root / "training_patches")
    valid_set = PatchSet.load(experiment_root / "validation_patches")
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


def inference_on_train_slides_pre_hnm() -> None:

    cp_path = (experiment_root / "patch_model").glob("*.ckpt")[0]
    classifier = PatchClassifier.load_from_checkpoint(
        checkpoint_path=cp_path, model=Backbone()
    )

    output_dir = experiment_root / "train_index"
    results_dir_name = "pre_hnm_results"
    heatmap_dir_name = "pre_hnm_heatmaps"

    train = PatchIndex.load(output_dir)

    # get number of classes
    just_patch_classes = remove_item_from_dict(train.labels, "background")
    num_classes = len(just_patch_classes)

    train_results = PatchIndexResults(train)

    for patchset in train_results:
        train_set = SlideDataset(patchset)
        probs_out = inference_on_slide(train_set, classifier, num_classes, 128, 80, 1)
        probs_df = pd.DataFrame(probs_out, columns=just_patch_classes)
        patchset = PatchSetResults(patchset, probs_df)
        patchset.save_csv(output_dir / results_dir_name)
        patchset.save_heatmap(output_dir / heatmap_dir_name)

    train_results.save(output_dir, results_dir_name, heatmap_dir_name)


def inference_on_valid_slides_pre_hnm() -> None:

    cp_path = (experiment_root / "patch_model").glob("*.ckpt")[0]
    classifier = PatchClassifier.load_from_checkpoint(
        checkpoint_path=cp_path, model=Backbone()
    )

    output_dir = experiment_root / "valid_index"
    results_dir_name = "pre_hnm_results"
    heatmap_dir_name = "pre_hnm_heatmaps"

    valid = PatchIndex.load(output_dir)

    # get number of classes
    just_patch_classes = remove_item_from_dict(valid.labels, "background")
    num_classes = len(just_patch_classes)

    valid_results = PatchIndexResults(valid)

    for patchset in valid_results:
        valid_set = SlideDataset(patchset)
        probs_out = inference_on_slide(valid_set, classifier, num_classes, 128, 80, 1)
        probs_df = pd.DataFrame(probs_out, columns=just_patch_classes)
        patchset = PatchSetResults(patchset, probs_df)
        patchset.save_csv(output_dir / results_dir_name)
        patchset.save_heatmap(output_dir / heatmap_dir_name)

    valid_results.save(output_dir, results_dir_name, heatmap_dir_name)


def hard_negative_mining_and_retrain() -> None:
    # read in original training patches
    train_set = PatchSet.load(experiment_root / "training_patches")

    # calculate false positives
    output_dir = experiment_root / "train_index"
    results_dir_name = "pre_hnm_results"
    train_results = SlidePatchIndexResults.load(Camelyon16, output_dir / results_dir_name)
    train_results = train_results.to_dataset()

    patch_dict = train_results.dataset.labels
    FP_mask = np.logical_and(train_results.patches_df['tumor'] > 0.5,
                             train_results.patches_df['label'] == patch_dict['tumor'])
    new_patches = train_results.patches_df[FP_mask]
    new_patches = new_patches.iloc[:, train_set.patches_df.columns]
    new_patches = PatchSet(train_results.dataset, train_results.size, train_results.level, new_patches)
    new_patches.save_patches(experiment_root / "training_patches")

    # combine together
    train_set.patches_df = pd.concat((train_set.patches_df, new_patches.patches_df))

    # prepare our data
    batch_size = 128
    valid_set = PatchSet(experiment_root / "validation_patches_hnm")
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
    test_patches = PatchIndex.for_dataset(test_data, tissue_detector, patch_finder)
    test_patches.save(experiment_root / "test_index")

    # index all the patches for the camelyon17 dataset
    train_data17 = camelyon17.training()
    test_data17 = camelyon17.testing()
    patch_finder = GridPatchFinder(6, 0, 256, 256)
    train_patches17 = PatchIndex.for_dataset(train_data17, tissue_detector, patch_finder)
    test_patches17 = PatchIndex.for_dataset(test_data17, tissue_detector, patch_finder)
    test_patches17.save(experiment_root / "test_index_17")

    # do the train validate split
    train17, valid17 = split_camelyon16(train_patches17, 0.7, seperate_slides=True)

    # save the train and valid patch indexes
    valid17.save(experiment_root / "valid_index_17")


def inference_on_valid_slides_post_hnm() -> None:

    cp_path = (experiment_root / "patch_model_hnm").glob("*.ckpt")[0]
    classifier = PatchClassifier.load_from_checkpoint(
        checkpoint_path=cp_path, model=Backbone()
    )

    output_dir = experiment_root / "valid_index"
    results_dir_name = "post_hnm_results"
    heatmap_dir_name = "post_hnm_heatmaps"

    valid = PatchIndex.load(output_dir)

    # get number of classes
    just_patch_classes = remove_item_from_dict(valid.labels, "background")
    num_classes = len(just_patch_classes)

    valid_results = PatchIndexResults(valid)

    for patchset in valid_results:
        valid_set = SlideDataset(patchset)
        probs_out = inference_on_slide(valid_set, classifier, num_classes, 128, 80, 1)
        probs_df = pd.DataFrame(probs_out, columns=just_patch_classes)
        patchset = PatchSetResults(patchset, probs_df)
        patchset.save_csv(output_dir / results_dir_name)
        patchset.save_heatmap(output_dir / heatmap_dir_name)

    valid_results.save(output_dir, results_dir_name, heatmap_dir_name)


def inference_on_test_slides_pre_hnm() -> None:

    cp_path = (experiment_root / "patch_model").glob("*.ckpt")[0]
    classifier = PatchClassifier.load_from_checkpoint(
        checkpoint_path=cp_path, model=Backbone()
    )

    output_dir = experiment_root / "test_index"
    results_dir_name = "pre_hnm_results"
    heatmap_dir_name = "pre_hnm_heatmaps"

    test = PatchIndex.load(output_dir)

    # get number of classes
    just_patch_classes = remove_item_from_dict(test.labels, "background")
    num_classes = len(just_patch_classes)

    test_results = PatchIndexResults(test)

    for patchset in test_results:
        test_set = SlideDataset(patchset)
        probs_out = inference_on_slide(test_set, classifier, num_classes, 128, 80, 1)
        probs_df = pd.DataFrame(probs_out, columns=just_patch_classes)
        patchset = PatchSetResults(patchset, probs_df)
        patchset.save_csv(output_dir / results_dir_name)
        patchset.save_heatmap(output_dir / heatmap_dir_name)

    test_results.save(output_dir, results_dir_name, heatmap_dir_name)


def inference_on_test_slides_post_hnm() -> None:

    cp_path = (experiment_root / "patch_model_hnm").glob("*.ckpt")[0]
    classifier = PatchClassifier.load_from_checkpoint(
        checkpoint_path=cp_path, model=Backbone()
    )

    output_dir = experiment_root / "test_index"
    results_dir_name = "post_hnm_results"
    heatmap_dir_name = "post_hnm_heatmaps"

    test = PatchIndex.load(output_dir)

    # get number of classes
    just_patch_classes = remove_item_from_dict(test.labels, "background")
    num_classes = len(just_patch_classes)

    test_results = PatchIndexResults(test)

    for patchset in test_results:
        test_set = SlideDataset(patchset)
        probs_out = inference_on_slide(test_set, classifier, num_classes, 128, 80, 1)
        probs_df = pd.DataFrame(probs_out, columns=just_patch_classes)
        patchset = PatchSetResults(patchset, probs_df)
        patchset.save_csv(output_dir / results_dir_name)
        patchset.save_heatmap(output_dir / heatmap_dir_name)

    test_results.save(output_dir, results_dir_name, heatmap_dir_name)


def inference_on_valid_slides_post_hnm_17() -> None:

    cp_path = (experiment_root / "patch_model_hnm").glob("*.ckpt")[0]
    classifier = PatchClassifier.load_from_checkpoint(
        checkpoint_path=cp_path, model=Backbone()
    )

    output_dir = experiment_root / "valid_index_17"
    results_dir_name = "post_hnm_results"
    heatmap_dir_name = "post_hnm_heatmaps"

    valid = PatchIndex.load(output_dir)

    # get number of classes
    just_patch_classes = remove_item_from_dict(valid.labels, "background")
    num_classes = len(just_patch_classes)

    valid_results = PatchIndexResults(valid)

    for patchset in valid_results:
        valid_set = SlideDataset(patchset)
        probs_out = inference_on_slide(valid_set, classifier, num_classes, 128, 80, 1)
        probs_df = pd.DataFrame(probs_out, columns=just_patch_classes)
        patchset = PatchSetResults(patchset, probs_df)
        patchset.save_csv(output_dir / results_dir_name)
        patchset.save_heatmap(output_dir / heatmap_dir_name)

    valid_results.save(output_dir, results_dir_name, heatmap_dir_name)


def inference_on_test_slides_post_hnm_17() -> None:

    cp_path = (experiment_root / "patch_model_hnm").glob("*.ckpt")[0]
    classifier = PatchClassifier.load_from_checkpoint(
        checkpoint_path=cp_path, model=Backbone()
    )

    output_dir = experiment_root / "test_index_17"
    results_dir_name = "post_hnm_results"
    heatmap_dir_name = "post_hnm_heatmaps"

    test = PatchIndex.load(output_dir)

    # get number of classes
    just_patch_classes = remove_item_from_dict(test.labels, "background")
    num_classes = len(just_patch_classes)

    test_results = PatchIndexResults(test)

    for patchset in test_results:
        test_set = SlideDataset(patchset)
        probs_out = inference_on_slide(test_set, classifier, num_classes, 128, 80, 1)
        probs_df = pd.DataFrame(probs_out, columns=just_patch_classes)
        patchset = PatchSetResults(patchset, probs_df)
        patchset.save_csv(output_dir / results_dir_name)
        patchset.save_heatmap(output_dir / heatmap_dir_name)

    test_results.save(output_dir, results_dir_name, heatmap_dir_name)


def inference_on_valid_slides_pre_hnm_17() -> None:

    cp_path = (experiment_root / "patch_model").glob("*.ckpt")[0]
    classifier = PatchClassifier.load_from_checkpoint(
        checkpoint_path=cp_path, model=Backbone()
    )

    output_dir = experiment_root / "valid_index_17"
    results_dir_name = "pre_hnm_results"
    heatmap_dir_name = "pre_hnm_heatmaps"

    valid = PatchIndex.load(output_dir)

    # get number of classes
    just_patch_classes = remove_item_from_dict(valid.labels, "background")
    num_classes = len(just_patch_classes)

    valid_results = PatchIndexResults(valid)

    for patchset in valid_results:
        valid_set = SlideDataset(patchset)
        probs_out = inference_on_slide(valid_set, classifier, num_classes, 128, 80, 1)
        probs_df = pd.DataFrame(probs_out, columns=just_patch_classes)
        patchset = PatchSetResults(patchset, probs_df)
        patchset.save_csv(output_dir / results_dir_name)
        patchset.save_heatmap(output_dir / heatmap_dir_name)

    valid_results.save(output_dir, results_dir_name, heatmap_dir_name)


def inference_on_test_slides_pre_hnm_17() -> None:

    cp_path = (experiment_root / "patch_model_hnm").glob("*.ckpt")[0]
    classifier = PatchClassifier.load_from_checkpoint(
        checkpoint_path=cp_path, model=Backbone()
    )

    output_dir = experiment_root / "test_index_17"
    results_dir_name = "pre_hnm_results"
    heatmap_dir_name = "pre_hnm_heatmaps"

    test = PatchIndex.load(output_dir)

    # get number of classes
    just_patch_classes = remove_item_from_dict(test.labels, "background")
    num_classes = len(just_patch_classes)

    test_results = PatchIndexResults(test)

    for patchset in test_results:
        test_set = SlideDataset(patchset)
        probs_out = inference_on_slide(test_set, classifier, num_classes, 128, 80, 1)
        probs_df = pd.DataFrame(probs_out, columns=just_patch_classes)
        patchset = PatchSetResults(patchset, probs_df)
        patchset.save_csv(output_dir / results_dir_name)
        patchset.save_heatmap(output_dir / heatmap_dir_name)

    test_results.save(output_dir, results_dir_name, heatmap_dir_name)


def calculate_lesion_level_results() -> None:

    results_pre_hnm = experiment_root / "valid_index" / "pre_hnm_results"
    results_post_hnm = experiment_root / "valid_index" / "post_hnm_results"

    valid_results_pre = SlidePatchIndexResults.load(Camelyon16, results_pre_hnm)
    valid_results_post = SlidePatchIndexResults.load(Camelyon16, results_pre_hnm)

    lesions_all_slides = pd.DataFrame(columns=["prob_score", "centre_row", "centre_col", "pixels", "filename"])

    for result_pre, results_post in zip(valid_results_pre, valid_results_post):
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
