import glob
import os
from abc import abstractmethod, ABCMeta


class Phase(metaclass=ABCMeta):
    @abstractmethod
    def feed_images(self):
        """
        Returns all images.
        :return:
        """

        return


class Training(Phase):

    def feed_images(self):
        pass

    def __init__(self, training_folder: str):
        super().__init__()
        self.tr_folder = training_folder
        self.training_images = None
        self.training_masks = None
        self.img_paths = None
        self.msk_paths = None
        self.__sync_metadata__()

    def __sync_metadata__(self):
        """
        Get's basic metadata for the class operation
        :return:
        """
        self.img_paths = glob.glob(os.path.join(self.tr_folder, "images/*"))
        self.msk_paths = glob.glob(os.path.join(self.tr_folder, "masks/*"))


class Testing(Phase):
    def __init__(self, testing_folder: str):
        pass
