from repath.preprocess.patching.patch_index import PatchIndexSet
from repath.data.datasets.camelyon16 import training, testing
from repath.preprocess.tissue_detection.tissue_detector import TissueDetectorOTSU
from repath.preprocess.patching.patch_finder import GridPatchFinder


def preprocessing() -> None:
    # import and load the Camelyon16 training set
    training_data = training()
    testing_data = testing()

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

