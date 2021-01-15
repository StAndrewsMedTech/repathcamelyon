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
