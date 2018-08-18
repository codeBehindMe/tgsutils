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
