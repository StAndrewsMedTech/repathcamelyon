import random
from typing import List

import torchvision.transforms.functional as TF

class Rotate(object):
    """ Create a rotate class """
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        image = TF.rotate(img, angle=self.angle)
        return image

class FlipRotate(object):
    """ Create a flip and rotate class """
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, img):
        """ Flips and then Rotates the image.
        
        Args:
            img (Image): The input image 

        Returns:
            image (Image): flipped and rotated imagg.

        """
        image = TF.hflip(img)
        image = TF.rotate(image, angle=self.angle)
        return image

class RandomRotateFromList(object):
    def __init__(self, angles: List[float]) -> None:
        super().__init__()
        self.angles = angles

    def __call__(self, img):
        # select angle
        angle = random.choice(self.angles)

        # apply angle
        image = TF.rotate(img, angle=angle)
        return image