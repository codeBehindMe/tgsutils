import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py


class Plot:

    @staticmethod
    def plot_image(img: np.ndarray, **kwargs) -> plt.Figure:
        """
        Plots an image.
        :param img: Image to be plotted.
        :param kwargs: Additional arguments to plot.
        :return:
        """

        fig = plt.figure()
        plt.imshow(img, **kwargs)
        return fig

    @staticmethod
    def plot_image_histogram(img: np.ndarray, **kwargs) -> plt.Figure:
        """
        Return the histogram of pixel intensities.
        :param img: Image to plot histogram of.
        :param kwargs: Additional arguments.
        :return:
        """
        flat = img.flatten()
        data = [go.Histogram(x=flat, **kwargs)]
        py.iplot(data, **kwargs)
