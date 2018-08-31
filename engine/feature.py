import numpy as np
from skimage import feature


def canny_filter(img: np.ndarray, **kwargs) -> np.ndarray:
    """
    Passes a canny edge detector filter over the image.
    :param img: Image to pass filter over.
    :param kwargs: Additional arguments to pass to canny filter.
    :return:
    """
    return feature.canny(img, **kwargs).astype(int)


def flip_image(img: np.ndarray, axis=1) -> np.ndarray:
    """
    This method flips an image along the vertical or horizontal axis.
    :param img: Image to be flipped.
    :param axis: Axis to be flipped on. axis= 0 vertical flip (up-down) and
    axis = 1 implies horizontal flip (left right).
    :return:
    """
    if axis == 0:
        return np.flipud(img)
    if axis == 1:
        return np.fliplr(img)
    else:
        raise ValueError("Invalid or unsupported axis.")
