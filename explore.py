import itertools
import os
from enum import Enum
from functools import wraps

import numpy as np

from phasing import Training, Testing


class Scope(Enum):
    TEST = 1
    TRAIN = 2
    VALIDATION = 3


class Explore:
    def __init__(self, training_folder: str, test_folder: str):
        self.tr_folder = training_folder
        self.ts_folder = test_folder
        self.scope = None
        self.core = Training(training_folder)  # FIXME : This is set so that IDE can get type hints. Set to None.
        self.__state__ = {"DEFAULT_SCOPE": None}

    @property
    def TEST(self):
        self.scope = Scope.TEST
        return self

    @property
    def TRAIN(self):
        self.scope = Scope.TRAIN
        return self

    def __load_scope__(self):
        """
        Loads the object corresponding to the current scope.
        :return:
        """
        if self.scope == Scope.TRAIN:
            self.core = Training(self.tr_folder)
            return
        if self.scope == Scope.TEST:
            self.core = Testing(self.ts_folder)
            return
        raise AttributeError(
            "Haven't set training or test! Please do so by calling TRAIN or TEST or set Default Scope.")

    def __reset_scope__(self):
        """
        Reset's the scope after a call has been made.
        :return:
        """
        self.scope = self.__state__['DEFAULT_SCOPE']

    def maintain_scope(f):
        @wraps(f)
        def wrapped(inst, *args, **kwargs):
            inst.__load_scope__()
            f_result = f(inst, *args, **kwargs)
            inst.__reset_scope__()
            return f_result

        return wrapped

    def set_default_scope(self, scope: Scope):
        """
        Set's the default scope for this instance so you don't have to call the SCOPE properties before calling
        methods.
        :param scope: Scope to change to.
        :return:
        """
        self.__state__['DEFAULT_SCOPE'] = scope
        self.scope = scope
        return self

    @maintain_scope
    def get_image_names(self, include_extension=False) -> [str]:
        """
        Returns a list of image names available.
        :return:
        """
        if include_extension:
            return [os.path.basename(file) for file in self.core.img_paths]
        else:
            return [os.path.splitext(os.path.basename(file))[0] for file in self.core.img_paths]

    @maintain_scope
    def get_mask_names(self, include_extension=False) -> [str]:
        """
        Returns a list of mask names. Only works if scope is training.
        :param include_extension:
        :return: List of mask names
        """
        if self.scope == Scope.TRAIN:
            if include_extension:
                return [os.path.basename(file) for file in self.core.msk_paths]
            else:
                return [os.path.splitext(os.path.basename(file))[0] for file in self.core.msk_paths]
        else:
            raise ValueError("Masks are only availbe for the training set")

    @maintain_scope
    def get_all_images(self) -> [np.ndarray]:
        """
        Returns all images as numpy arrays
        :return:
        """

        for img in self.core.get_all_images():
            return img

    @maintain_scope
    def get_image_by_index(self, index):
        """
        Returns the image at the specified index.
        :param index:
        :return:
        """
        return next(itertools.islice(self.core.get_all_images(), index, index + 1))

    @maintain_scope
    def get_image_by_name(self, name: str) -> np.ndarray:
        return self.get_image_by_index(self.get_image_names().index(os.path.splitext(name)[0]))

    @maintain_scope
    def get_image_sample(self, sample_size: int, random=False) -> [np.ndarray]:
        """
        Returns a subsample of images at the sample size. Can specify random to randomise the sample or preserve order.
        :param sample_size: Size of the sample required.
        :param random: Randomise or not.
        :return:
        """
        sample_images = []
        if random:
            indexes = sorted(np.random.randint(0, self.core.iterable_size - 1, sample_size))
            i = 0
            try:
                for img in self.core.get_all_images():
                    if i == indexes[0]:
                        sample_images.append(img)
                        indexes.pop(0)
                        i += 1
                    else:
                        i += 1
                        continue
            except IndexError:
                pass

        else:
            imgs = self.core.get_all_images()
            return [next(imgs) for x in range(sample_size)]
        return sample_images

    @maintain_scope
    def get_images(self, **kwargs) -> iter:
        """
        Returns images based on what's required. Calling this method with no keyword arguments returns all images as
        numpy arrays.
        :return:
        """
        try:
            return self.get_image_by_name(kwargs['name'])
        except KeyError:
            pass
        try:
            return self.get_image_by_index(kwargs['index'])
        except KeyError:
            pass
        try:
            return self.get_image_sample(kwargs['sample'], False)
        except KeyError:
            pass
        try:
            return self.get_image_sample(kwargs['random_sample'], True)
        except KeyError:
            pass
        return self.get_all_images()


if __name__ == '__main__':
    x = Explore("train", "test")
    # print(x.scope)
    # print(x.TRAIN.get_image_names())
    # print(x.scope)
    # print(x.TRAIN.get_mask_names())
    #
    # print(x.TRAIN.get_all_images()[0])

    print(x.TRAIN.get_image_sample(3, False).__len__())
