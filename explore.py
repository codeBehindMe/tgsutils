from enum import Enum
from functools import wraps

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
        raise AttributeError("Havent set training or test! Please do so by calling TRAIN or TEST or set Default Scope.")

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
    def get_image_names(self) -> [str]:
        """
        Returns a list of image names available.
        :return:
        """
        result = self.core.image_names
        return result


if __name__ == '__main__':
    x = Explore("train", "test")
    print(x.scope)
    print(x.TRAIN.get_image_names())
    print(x.scope)
