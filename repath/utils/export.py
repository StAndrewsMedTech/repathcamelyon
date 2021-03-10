import cv2
import numpy as np
from PIL import Image


def convert_mask_to_contours_json(im_resize, label):
    # get contours of binary mask
    contours, hierarchy  = cv2.findContours(np.array(im_resize, dtype=np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    json_points = []

    for ct in range(hierarchy.shape[1]):
        got_parent = hierarchy[0, ct, 3] >= 0
        if not got_parent:
            child_mask = hierarchy[0, :, 3] == ct
            got_kids = np.sum(child_mask) > 0
            if got_kids:
                polygon1 = contours[ct]
                pts = []
                for pt in range(polygon1.shape[0]):
                    pnt = polygon1[pt][0].tolist()
                    pts.append(pnt)
                json_polygon = [pts]
                kids = np.array(contours, dtype=object)[np.arange(len(contours))[child_mask]]
                for kid in kids:
                    pts = []
                    for pt in range(kid.shape[0]):
                        pnt = kid[pt][0].tolist()
                        pts.append(pnt)
                    json_polygon.append(pts)
                json_points.append(json_polygon)
            else:
                pts = []
                for pt in range(contours[ct].shape[0]):
                    pnt = contours[ct][pt][0].tolist()
                    pts.append(pnt)
                json_points.append([pts])
                
    json_dict = {label: json_points}
    return json_dict



def convert_mask_to_json(binary_mask: np.ndarray, label: str, level_in: int, level_out: int = 5):
    # convert to PIL image
    im_out = Image.fromarray(np.array(binary_mask*255, dtype=np.uint8))

    # calculate new image size
    level_diff = level_in - level_out
    size_diff = 2 ** level_diff
    new_size = (im_out.size[0] * size_diff, im_out.size[1] * size_diff)

    # resize image
    im_resize = im_out.resize(new_size, Image.BOX)
    
    # get json dictionary of contours
    json_dict = convert_mask_to_contours_json(im_resize, label)

    return json_dict
