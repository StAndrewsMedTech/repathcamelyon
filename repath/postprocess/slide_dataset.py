from typing import Tuple

from PIL import Image
from torch.utils.data import Dataset

from repath.data.slides.slide import Region


class SlideDataset(Dataset):
    def __init__(self, ps: 'PatchSet', transform = None, augments = None) -> None:
        super().__init__()
        self.ps = ps
        self.slide = ps.dataset.slide_cls(ps.abs_slide_path)
        self.transform = transform
        self.augments = augments

    def open_slide(self):
        self.slide.open()

    def close_slide(self):
        self.slide.close()

    def to_patch(self, p: tuple) -> Image:
        region = Region.patch(p.x, p.y, self.ps.patch_size, self.ps.level)
        image = self.slide.read_region(region)
        image = image.convert('RGB')
        return image

    def __len__(self):
        return len(self.ps)

    def __getitem__(self, idx):
        patch_info = self.ps[idx]
        image = self.to_patch(patch_info)
        label = patch_info.label
        if self.transform is not None:
            if self.augments is None:
                image = self.transform(image)
            else: 
                augment = augments[patch_info.transform - 1]
                image = augment(image)
                image = self.transform(image)
        return image, label
