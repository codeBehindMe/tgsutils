import glob
import os
from enum import Enum


class Scope(Enum):
    TEST = 1
    TRAIN = 2
    VALIDATION = 3


class Training:

    def __init__(self, training_folder):
        self.tr_folder = training_folder
        self.training_images = None
        self.training_masks = None
        self.image_names = None
        self.img_paths = None

    def load_images(self):
        path = os.path.join(self.training_images, "images/*")
        self.img_paths = glob.glob(path)
        self.image_names = map(os.path.basename, self.img_paths)


class Explore:
    def __init__(self, training_folder: str, test_folder: str):
        self.tr_folder = training_folder
        self.ts_folder = test_folder
        self.scope = None

    @property
    def TEST(self):
        self.scope = Scope.TEST
        return self

    @property
    def TRAIN(self):
        self.scope = Scope.TRAIN
        return self

    def random_method(self):
        print(self.scope)


if __name__ == '__main__':
    x = Explore("hello", "mellow")
    x.random_method()
    x.TEST.random_method()
