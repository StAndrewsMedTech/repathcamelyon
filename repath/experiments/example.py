from repath.utils.paths import project_root
from repath.preprocess.patching import PatchIndex
import repath.data.datasets.camelyon16 as camelyon16
from repath.preprocess.tissue_detection import TissueDetectorOTSU
from repath.preprocess.patching import GridPatchFinder


"""
Global stuff
"""

experiment_root = project_root() / 'experiments' / 'wang'
tissue_detector = TissueDetectorOTSU()

class PatchClassifier(pl.LightningModule):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

    def cross_entropy_loss(self, logits, labels):
        return F.nll_loss(logits, labels)

    def training_step(self, batch, batch_idx):
        x, y = train_batch
        logits = self.model(x)
        x = torch.log_softmax(x, dim=1)
        loss = self.cross_entropy_loss(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = train_batch
        logits = self.model(x)
        x = torch.log_softmax(x, dim=1)
        loss = self.cross_entropy_loss(logits, y)
        self.log('val_loss', loss)

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
    train, valid = split(train_patches, 0.7)
    train_samples = sample(train)
    valid_samples = sample(valid)

    # save out all the patches
    train_samples.save_patches(experiment_root / "training_patches")
    valid_samples.save_patches(experiment_root / "training_patches")


def train_patch_classifier() -> None:
    # prepare our data 
    batch_size = 128
    train_set = ImageFolder(experiment_root / "training_patches")
    valid_set = ImageFolder(experiment_root / "validation_patches")
    train_loader = DataLoader(train_data, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, batch_size=batch_size)

    
    model = Backbone()
    classifier = PatchClassifier(model)
    trainer = pl.Trainer()
    trainer.fit(classifier, train_dataloader=train_loader, val_dataloaders=valid_loader)


def postprocessing() -> None:
    pass


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
