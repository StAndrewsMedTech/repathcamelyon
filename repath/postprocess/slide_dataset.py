from typing import Tuple
from torch import Dataset

from PIL import Image

from repath.preprocess.patching import PatchSet
from repath.utils.geometry import Region

class SlideDataset(Dataset):
    def __init__(self, patchset: PatchSet, transform=None) -> None:
        def to_sample(p: tuple) -> Tuple[Region, Image]:
            region = Region.patch(p.x, p.y, patchset.patch_size, patchset.level)
            image = slide.read_region(region)
            return region, image
        
        self.transform = transform
        with patchset.open_slide() as slide:
            self.samples = [to_sample(p) for p in patchset]

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        image, label = self.samples[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
