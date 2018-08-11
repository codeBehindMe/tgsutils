import glob
import os
from abc import abstractmethod


class Phase:
    @abstractmethod
    def feed_images(self):
        """
        Returns all images.
        :return:
        """

        return


class Training(Phase):

    def __init__(self, training_folder: str):
        super().__init__()
        self.tr_folder = training_folder
        self.training_images = None
        self.training_masks = None
        self.image_names = None
        self.img_paths = None
        self.sync_image_metadata()

    def sync_image_metadata(self):
        path = os.path.join(self.tr_folder, "images/*")
        self.img_paths = glob.glob(path)
        self.image_names = list(map(os.path.basename, self.img_paths))


class Testing(Phase):
    def __init__(self, testing_folder: str):
        pass
