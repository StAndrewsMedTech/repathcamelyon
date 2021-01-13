from typing import Tuple

from PIL import Image
from torch.utils.data import Dataset

from repath.data.slides.slide import Region


class SlideDataset(Dataset):
    def __init__(self, patchset: 'SlidePatchSet', transform=None) -> None:
        #def to_sample(p: tuple) -> Tuple[Region, Image]:
        #    region = Region.patch(p.x, p.y, patchset.patch_size, patchset.level)
        #    image = slide.read_region(region)
        #    return region, image

        def to_patch(p: tuple) -> Image:
            region = Region.patch(p.x, p.y, patchset.patch_size, patchset.level)
            image = slide.read_region(region)
            image = image.convert('RGB')
            return image
        
        self.transform = transform
        with patchset.open_slide() as slide:
            # self.samples = [to_sample(p) for p in patchset]
            self.samples = [(to_patch(p), p.label) for p in patchset]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label = self.samples[idx]
        if self.transform is not None:
            image = self.transform(image)
        return image, label
