from typing import List, Dict

import cv2
import numpy as np

from repath.utils.geometry import PointF, Shape

annotation_types = ["Dot", "Polygon", "Spline", "Rectangle"]


class Annotation:
    def __init__(
        self, name: str, annotation_type: str, label: str, vertices: List[PointF]
    ):
        assert annotation_type in annotation_types
        self.name = name
        self.type = annotation_type
        self.label = label
        self.coordinates = vertices

    def draw(self, image: np.array, labels: Dict[str, int], factor: float):
        """Renders the annotation into the image.

        Args:
            image (np.array): Array to write the annotations into, must have dtype float.
            labels (Dict[str, int]): The value to write into the image for each type of label.
            factor (float): How much to scale (by divison) each vertex by.
        """
        fill_colour = labels[self.label]
        vertices = np.array(self.coordinates) / factor
        vertices = vertices.astype(np.int32)
        cv2.fillPoly(image, [vertices], (fill_colour))


def render_annotations(
    annotations: List[Annotation],
    factor: float,
    shape: Shape,
    labels: Dict[str, int],
    labels_order: List[str],
    background_label: str,
) -> np.array:
    annotations = sorted(annotations, key=lambda a: labels_order.index(a.label))
    image = np.full(shape, labels[background_label], dtype=float)
    for a in annotations:
        a.draw(image, labels, factor)
    return image.astype("int")
