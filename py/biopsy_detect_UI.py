from wholeslidedata.image.wholeslideimage import WholeSlideImage
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

from py.helpers import BARRET_ROOT


BIOPSY_DETECT_TEST_DIR = os.path.join(BARRET_ROOT, "Luuk_biopsy-detection", "test_data")

def get_slide_at_idx(idx, root_dir=BIOPSY_DETECT_TEST_DIR):
    slidepaths = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.tiff')]
    slidepath = slidepaths[idx]
    try:
        wsi = WholeSlideImage(slidepath)
    except Exception as e:
        print(e)
        return None, slidepath
    return wsi, slidepath


def plot_clean(img, cmap=None, ax=None):
    if ax is None:
        plt.imshow(img, cmap=cmap)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad=0)
        plt.show()
    else:
        ax.imshow(img, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal') # to make sure the image is not stretched


def get_contoured_img(img, contours, boxes, thickness=1, color_by_idx=None, put_text=False):
    def thick_fn(value):
        return max(int(value * np.max(img.shape[:2]) / 1000 * thickness), 1)
    
    for i, (x, y, w, h) in enumerate(boxes):
        if color_by_idx is None or i not in color_by_idx:
            contour_color = (0, 0, 255)
            box_color = (0, 0, 0)
        else:
            contour_color = color_by_idx[i]['contour']
            box_color = color_by_idx[i]['box']

        img = cv2.drawContours(img.astype(np.uint8), [contours[i]], -1, contour_color, thick_fn(4))
        if box_color is not None:
            cv2.rectangle(img, (x, y), (x+w, y+h), box_color, thick_fn(1.5))
            # Put cross in center
            cross_size = thick_fn(4)
            cv2.line(img, (x+w//2 - cross_size, y+h//2), (x+w//2 + cross_size, y+h//2), (0, 0, 0), thick_fn(1.5))
            cv2.line(img, (x+w//2, y+h//2 - cross_size), (x+w//2, y+h//2 + cross_size), (0, 0, 0), thick_fn(1.5))

            if put_text:
                # Put text of contour area in top left corner
                area = cv2.contourArea(contours[i])
                cv2.putText(img, str(int(area)), (x, y+h-thick_fn(4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5*thick_fn(1), (0, 0, 0), thick_fn(2))
    return img


def make_color_by_idx(too_small_indices, 
        bg_indices, covered_indices, edge_hit_indices, colors=[
            (255, 0, 0), (0, 0, 0), (255, 0, 255), (255, 255, 0)
        ]):
    """
    Makes a dictionary mapping contour index to a color.

    Parameters:
        too_small_indices: list
            List of indices of contours that are too small
        bg_indices: list
            List of indices of contours that surround background pixels
        covered_indices: list
            List of indices of contours that cover the patch
        edge_hit_indices: list
            List of indices of contours that have more than 2 edge hits
        colors: list
            List of colors to use for each type of contour in the order:
            too small, background, covered, edge hit

    Returns:
        color_by_idx: dictionary
            Dictionary mapping contour index to a color
    """
    color_by_idx = {}
    for i in too_small_indices:
        color_by_idx[i] = {'contour': colors[0], 'box': None}
    for i in bg_indices:
        color_by_idx[i] = {'contour': colors[1], 'box': None}
    for i in covered_indices:
        color_by_idx[i] = {'contour': colors[2], 'box': None}
    for i in edge_hit_indices:
        color_by_idx[i] = {'contour': colors[3], 'box': None}
    return color_by_idx