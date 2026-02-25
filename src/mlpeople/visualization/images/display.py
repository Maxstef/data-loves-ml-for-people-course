import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


def row_to_image(row, shape=(28, 28), normalize=False):
    """
    Convert a pandas Series or 1D array to a 2D image.

    Args:
        row: pandas Series or 1D numpy array
        shape: tuple, image shape (height, width)
        normalize: bool, if True scales pixel values to [0,1]

    Returns:
        2D numpy array
    """
    img = row.to_numpy() if hasattr(row, "to_numpy") else np.array(row)
    img = img.reshape(shape)
    if normalize:
        img = img.astype(float) / img.max()
    return img


def show_image_matplotlib(
    row, shape=(28, 28), normalize=False, hide_axis=True, cmap="gray"
):
    """
    Display a 2D image using Matplotlib.
    """
    img = row_to_image(row, shape, normalize)
    plt.imshow(img, cmap=cmap, interpolation="nearest")
    if hide_axis:
        plt.axis("off")
    plt.show()


def show_image_plotly(
    row, shape=(28, 28), normalize=False, hide_colorbar=True, cmap="gray"
):
    """
    Display a 2D image using Plotly.
    """
    img = row_to_image(row, shape, normalize)
    fig = px.imshow(img, color_continuous_scale=cmap)
    if hide_colorbar:
        fig.update_layout(coloraxis_showscale=False, yaxis=dict(scaleanchor="x"))
    fig.show()


def print_image_ascii(row, shape=(28, 28), scale_to_9=True, pad_width=2):
    """
    Print a 2D image as ASCII-like numbers.

    Args:
        row: pandas Series or 1D array
        shape: tuple of image shape
        scale_to_9: scale pixel values to 0-9 for compact printing
        pad_width: number of spaces per number
    """
    img = row_to_image(row, shape)
    if scale_to_9:
        img = (img / img.max() * 9).astype(int)

    lines = []
    for r in img:
        line = "".join(str(v).rjust(pad_width, " ") for v in r)
        lines.append(line)
    print("\n".join(lines))
