import glob
import os
from abc import abstractmethod, ABCMeta

import cv2


class Phase(metaclass=ABCMeta):
    @abstractmethod
    def get_all_images(self):
        """
        Returns all images.
        :return:
        """

        return


class Training(Phase):

    def get_all_images(self):
        """
        Returns all images as numpy matricies.
        :return:
        """
        for i in self.img_paths:
            yield cv2.imread(i)

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
