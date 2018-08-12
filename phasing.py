import glob
import os
from abc import abstractmethod, ABCMeta

import cv2
import numpy as np


class Phase(metaclass=ABCMeta):
    @abstractmethod
    def get_all_images(self):
        """
        Returns all images.
        :return:
        """
        pass

    @abstractmethod
    def get_image_by_name(self, name: str):
        """
        Returns the image for the specified name.
        :param name:
        :return:
        """
        pass

    @abstractmethod
    def get_all_masks(self):
        """
        Returns all masks.
        :return:
        """
        pass

    @abstractmethod
    def get_mask_by_name(self, name: str) -> np.ndarray:
        """
        Returns the mask for the specified name.
        :param name:
        :return:
        """
        pass


class Training(Phase):

    def get_image_by_name(self, name: str) -> np.ndarray:
        for path in self.img_paths:
            if os.path.splitext(os.path.basename(path)[0]) == name:
                return cv2.imread(path)

    def get_all_images(self) -> iter:
        for path in self.img_paths:
            yield cv2.imread(path)

    def get_all_masks(self) -> iter:
        for path in self.msk_paths:
            yield cv2.imread(path)

    def get_mask_by_name(self, name: str) -> np.ndarray:
        for path in self.msk_paths:
            if os.path.splitext(os.path.basename(path)[0]) == name:
                return cv2.imread(path)

    def __init__(self, training_folder: str):
        super().__init__()
        self.tr_folder = training_folder
        self.training_images = None
        self.training_masks = None
        self.img_paths = None
        self.msk_paths = None
        self.iterable_size = None
        self.__sync_metadata__()

    def __sync_metadata__(self):
        """
        Get's basic metadata for the class operation
        :return:
        """
        self.img_paths = glob.glob(os.path.join(self.tr_folder, "images/*"))
        self.msk_paths = glob.glob(os.path.join(self.tr_folder, "masks/*"))
        self.iterable_size = len(self.img_paths)


class Testing(Phase):
    def __init__(self, testing_folder: str):
        pass
