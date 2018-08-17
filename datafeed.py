import itertools
import operator
import os
from enum import Enum
from functools import wraps

import numpy as np

from phasing import Training, Testing


class Scope(Enum):
    TEST = 1
    TRAIN = 2
    VALIDATION = 3


class DataFeed:
    def __init__(self, training_folder: str, test_folder: str):
        self.tr_folder = training_folder
        self.ts_folder = test_folder
        self.scope = None
        # FIXME : This is set so that IDE can get type hints. Set to None.
        self.core = Training(training_folder)
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
            "Haven't set training or test! Please do so by calling TRAIN or "
            "TEST or set default scope.")

    def __reset_scope__(self):
        """
        Reset's the scope after a call has been made.
        :return:
        """
        self.scope = self.__state__['DEFAULT_SCOPE']

    def maintain_scope(f):
        """
        Decorator to maintain scope when endpoint methods are called.
        :return:
        """

        @wraps(f)
        def wrapped(inst, *args, **kwargs):
            inst.__load_scope__()
            f_result = f(inst, *args, **kwargs)
            inst.__reset_scope__()
            return f_result

        return wrapped

    def set_default_scope(self, scope: Scope):
        """
        Set's the default scope for this instance so you don't have to
        call the SCOPE properties before calling methods.
        :param scope: Scope to change to.
        :return:
        """
        self.__state__['DEFAULT_SCOPE'] = scope
        self.scope = scope
        return self

    @maintain_scope
    def get_names(self):
        return self.get_image_names()

    @maintain_scope
    def get_image_names(self, include_extension=False) -> [str]:
        """
        Returns a list of image names available.
        :return:
        """
        if include_extension:
            return [os.path.basename(file) for file in self.core.img_paths]
        else:
            return [os.path.splitext(os.path.basename(file))[0] for file in
                    self.core.img_paths]

    @maintain_scope
    def get_mask_names(self, include_extension=False) -> [str]:
        """
        Returns a list of mask names. Only works if scope is training.
        :param include_extension:
        :return: List of mask names
        """
        if self.scope == Scope.TRAIN:
            if include_extension:
                return [os.path.basename(file) for file in
                        self.core.msk_paths]
            else:
                return [os.path.splitext(os.path.basename(file))[0] for file
                        in self.core.msk_paths]
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
        return next(
            itertools.islice(self.core.get_all_images(), index, index + 1))

    @maintain_scope
    def get_image_by_name(self, name: str) -> np.ndarray:
        return self.get_image_by_index(
            self.get_image_names().index(os.path.splitext(name)[0]))

    @maintain_scope
    def get_image_sample(self, sample_size: int, random=False) -> [
        np.ndarray]:
        """
        Returns a subsample of images at the sample size. Can specify random
        to randomise the sample or preserve order.
        :param sample_size: Size of the sample required.
        :param random: Randomise or not.
        :return:
        """
        if random:
            indexes = sorted(np.random.randint(0, self.core.iterable_size - 1,
                                               sample_size))
            return [self.core.get_image_by_name(name) for name in
                    operator.itemgetter(*indexes)(self.get_image_names())]
        else:
            imgs = self.core.get_all_images()
            return [next(imgs) for x in range(sample_size)]

    @maintain_scope
    def get_all_masks(self):
        """
        Returns all masks as numpy arrays.
        :return:
        """
        for msk in self.core.get_all_masks():
            return msk

    @maintain_scope
    def get_mask_by_index(self, index):
        """
        Returns the mask at the specified index.
        :param index:
        :return:
        """
        return next(
            itertools.islice(self.core.get_all_masks(), index, index + 1))

    @maintain_scope
    def get_mask_by_name(self, name: str) -> np.ndarray:
        """
        Returns image of given name.
        :param index:
        :return:
        """
        return self.get_mask_by_index(
            self.get_mask_names().index(os.path.splitext(name)[0]))

    @maintain_scope
    def get_mask_sample(self, sample_size: int, random=False) -> [np.ndarray]:
        """
        Returns a sample of the masks of size sample_size. Can specify random
        to randomise the sample or preserve order.
        :param sample_size:
        :param random:
        :return:
        """
        if random:
            indexes = sorted(np.random.randint(0, self.core.iterable_size - 1,
                                               sample_size))
            return [self.core.get_mask_by_name(name) for name in
                    operator.itemgetter(*indexes)(self.get_mask_names())]
        else:
            masks = self.core.get_all_masks()
            return [next(masks) for i in range(sample_size)]

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

    @maintain_scope
    def get_masks(self, **kwargs) -> np.ndarray:
        """
        Returns masks based on what's required. Calling this method with no arguments returns all masks.
        :param kwargs:
        :return:
        """
        try:
            return self.get_mask_by_name(kwargs['name'])
        except KeyError:
            pass
        try:
            return self.get_mask_by_index(kwargs['index'])
        except KeyError:
            pass
        try:
            return self.get_mask_sample(kwargs['sample'], False)
        except KeyError:
            pass
        try:
            return self.get_mask_sample(kwargs['random_sample'], True)
        except KeyError:
            pass
        return self.get_all_masks()

    @maintain_scope
    def get_all(self) -> zip:
        """
        Returns tuples of images and masks for all available masks.
        :return:
        """
        # FIXME: Zip iterator not working ***
        if self.scope != Scope.TRAIN:
            raise ValueError("Can only call this from TRAIN scope.")
        else:
            items = zip(self.TRAIN.get_all_images(),
                        self.TRAIN.get_all_masks())
            return [(img, val) for img, val in items]

    @maintain_scope
    def get_by_index(self, index: int) -> tuple:
        """
        Returns a tuple of image and mask for the specified index.
        :param index:
        :return:
        """
        if self.scope != Scope.TRAIN:
            raise ValueError("Can only call this from TRAIN scope.")
        else:
            return self.TRAIN.get_image_by_index(
                index=index), self.TRAIN.get_mask_by_index(index=index)

    @maintain_scope
    def get_by_name(self, name: str) -> tuple:
        """
        Returns a tuple of image and mask for the specified name.
        :param name: Name of the image.
        :return:
        """
        if self.scope != Scope.TRAIN:
            raise ValueError("Can only call this from TRAIN scope.")
        else:
            return self.TRAIN.get_image_by_name(
                name), self.TRAIN.get_mask_by_name(name)

    @maintain_scope
    def get_sample(self, sample_size: int, random=False) -> tuple:
        """
        Get's a random sample of images and their corresponding masks. Can
        specify random bool to make sure the feed is shuffled.
        :param sample_size: Size of sample required.
        :param random: Shuffle parameter.
        :return:
        """
        return


if __name__ == '__main__':
    x = DataFeed("train", "test")
    # print(x.scope)
    # print(x.TRAIN.get_image_names())
    # print(x.scope)
    # print(x.TRAIN.get_mask_names())
    #
    # print(x.TRAIN.get_all_images()[0])
    print(x.TRAIN.get_all()[0][0].shape)
