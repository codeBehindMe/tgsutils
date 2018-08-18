import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py


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


def plot_image_histogram(img: np.ndarray, **kwargs) -> plt.Figure:
    """
    Return the histogram of pixel intensities.
    :param img: Image to plot histogram of.
    :param kwargs: Additional arguments.
    :return:
    """
    flat = img.flatten()
    data = [go.Histogram(x=flat, **kwargs)]
    return py.iplot(data)


def plot_multi_image_column_wise(*args, **kwargs) -> plt.Figure:
    """
    This plots multiple images in a column wise fashion.
    :param args: List of images.
    :param kwargs: additional keyword arguments to plot options.
    :return:
    """
    _len = len(args)

    fig = plt.figure(figsize=kwargs.get("figsize", (5, 5)))
    for i in range(_len):
        plt.subplot(1, _len, i + 1)
        plt.imshow(args[i])

    return fig
