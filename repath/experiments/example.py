import pandas as pd
from pytorch_lightning.metrics import Accuracy

from repath.utils.paths import project_root
from repath.utils.convert import remove_item_from_dict
from repath.preprocess.patching import PatchIndex
import repath.data.datasets.camelyon16 as camelyon16
from repath.preprocess.tissue_detection import TissueDetectorOTSU
from repath.preprocess.patching import GridPatchFinder
from repath.patch_classification.models.simple import Backbone
from repath.postprocess.slide_dataset import SlideDataset
from repath.postprocess.prediction import inference_on_slide
from repath.postprocess.patch_index_results import PatchSetResults, PatchIndexResults


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
    train, valid = split(train_patches, 0.7, seperate_slides=True)
    train_samples = sample(train, 700000)
    valid_samples = sample(valid, 300000)

    # save the train and valid patch indexes
    train.save(experiment_root / "train_index")
    valid.save(experiment_root / "valid_index")

    # save out all the patches
    train_samples.save_patches(experiment_root / "training_patches")
    valid_samples.save_patches(experiment_root / "validation_patches")


def train_patch_classifier() -> None:
    # prepare our data
    batch_size = 128
    train_set = ImageFolder(experiment_root / "training_patches")
    valid_set = ImageFolder(experiment_root / "validation_patches")
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


def inference_on_valid_slides() -> None:

    cp_path = (experiment_root / "patch_model").glob("*.ckpt")[0]
    classifier = PatchClassifier.load_from_checkpoint(
        checkpoint_path=cp_path, model=Backbone()
    )

    valid = PatchIndex.load(experiment_root / "valid_index")

    # get number of classes
    just_patch_classes = remove_item_from_dict(valid.labels, "background")
    num_classes = len(just_patch_classes)

    output_dir = experiment_root / "valid_index"
    results_dir_name = "pre_hnm_results"
    heatmap_dir_name = "pre_hnm_heatmaps"
    valid_results = PatchIndexResults(valid)

    for patchset in valid:
        valid_set = SlideDataset(patchset)
        probs_out = inference_on_slide(valid_set, classifier, num_classes, 128, 80, 1)
        probs_df = pd.DataFrame(probs_out, columns=just_patch_classes)
        patchset_results = PatchSetResults(patchset, probs_df)
        patchset_results.save_csv(output_dir / results_dir_name)
        patchset.heatmap.save_heatmap(output_dir / heatmap_dir_name)

    valid_results.save(output_dir, results_dir_name, heatmap_dir_name)


def slide_training() -> None:
    pass


# slides sampling - train and validation

# patch sampling - within train or validation, balance the classes over the slides - this is the same as preprocessing

# train patch classifier

# hard negative mining
#   perdict for every patch in slide sampling - train
#   take the false positive and add them back into the patch sampling
#   retain the patch classifier

# using the slide sampling validation set, predict a probability for every patch to create a heat map

# there are patch results (can use slides and patches sets), slide results, lesion level results, patient level results

steps = [preprocessing, patch_training]
