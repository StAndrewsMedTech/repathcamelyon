from pathlib import Path
from typing import List
import xml.etree.ElementTree as ET

from repath.data.annotations.annotation import Annotation


def annotation_from_tag(tag: ET.Element) -> Annotation:
    # get the attributes
    name = tag.attrib["Name"]
    group = tag.attrib["PartOfGroup"]
    annotation_tag = tag.attrib["Type"]
    coordinate_tags = tag.find("Coordinates")

    # groups Tumor, _0 and _1 are tumor annoations and group _2 are normal annoations
    assert group in ["Tumor", "_0", "_1", "_2"], "Unknown annoation group encountered."
    label = "tumor" if group in ["Tumor", "_0", "_1"] else "normal"

    # parse the coordinate to a list of lists with two floats
    vertices = [(float(c.attrib["X"]), float(c.attrib["Y"])) for c in coordinate_tags]

    # pass the data to the annotation factory
    return Annotation(name, annotation_tag, label, vertices)


def load_annotations(xml_file_path: Path) -> List[Annotation]:
    # if the path is empty or a dir then return an empty annotations list
    # TODO: Make sure this requirement is stated in the requirements for
    # load_annotations functions
    if not xml_file_path.is_file():
        return []

    # find all the annotation tags in the xml document
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    tags = root.find("Annotations")

    # get the type and colour properties and coordinated for each annotation
    annotations = [annotation_from_tag(tag) for tag in tags]
    annotations = [a for a in annotations if a]  # remove None values

    return annotations
