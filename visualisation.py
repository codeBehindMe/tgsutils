import matplotlib.pyplot as plt
import numpy as np


class Plot:

    @staticmethod
    def plot_image(img: np.ndarray, **kwargs) -> plt.Figure:
        """
        Plots an image.
        :param img:
        :return:
        """

        fig = plt.figure()
        plt.imshow(img, **kwargs)
        return fig
