from os import POSIX_FADV_WILLNEED
from repath.preprocess.patching import PatchIndexSet
import repath.data.datasets.camelyon16 as camelyon16
from repath.preprocess.tissue_detection.tissue_detector import TissueDetectorOTSU
from repath.preprocess.patching.patch_finder import GridPatchFinder


def preprocessing() -> None:
    # import and load the Camelyon16 training set
    training_data = camelyon16.training()
    testing_data = camelyon16.testing()

    # define a tissue detector and the patch finder for creating the patch index
    tissue_detector = TissueDetectorOTSU()
    patch_finder = GridPatchFinder(6, 0, 256, 256)

    # use the dataset to generate a patch index set for the training data
    training_patches = PatchIndexSet(training_data, tissue_detector, patch_finder)
    testing_patches = PatchIndexSet(testing_data, tissue_detector, patch_finder)

    # create a recipe that balances the number of patches each label in each patch set


    # save the patches to the cache directory


def patch_training() -> None:
    pass


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

