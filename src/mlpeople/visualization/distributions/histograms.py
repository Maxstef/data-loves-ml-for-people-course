import plotly.express as px
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
import numpy as np
import matplotlib.pyplot as plt


def px_histogram(
    df=None,
    x=None,
    marginal="box",
    color_discrete_sequence=None,
    color=None,
    title=None,
    show=True,
    **kwargs,
):
    # Build title
    if title is None:
        title = x

        if title and color:
            title = f"{title} per {color}"

    fig = px.histogram(
        df,
        x=x,
        marginal=marginal,
        color=color,
        color_discrete_sequence=color_discrete_sequence,
        title=title,
        **kwargs,
    )

    fig.update_layout(bargap=0.1)

    if show:
        fig.show()

    return fig


def px_histogram_comparison(
    df, x, color, title_left, title_right, histnorm_right="percent"
):
    fig = make_subplots(rows=1, cols=2, subplot_titles=[title_left, title_right])

    fig_left = px_histogram(
        df, x=x, color=color, marginal=None, title=title_left, show=False
    )

    fig_right = px_histogram(
        df,
        x=x,
        color=color,
        marginal=None,
        title=title_right,
        histnorm=histnorm_right,
        show=False,
    )

    for trace in fig_left.data:
        fig.add_trace(trace, row=1, col=1)

    for trace in fig_right.data:
        fig.add_trace(trace, row=1, col=2)

    fig.update_layout(bargap=0.1)

    categories = df[x].unique()
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=categories,
        showticklabels=True,
        row=1,
        col=2,
    )

    fig.show()
    return fig


def plt_histogram(
    df, col, color_col, labels=None, bins=30, density=True, kde=True, colors=None
):
    if labels is None:
        labels = {}

    # Default colors: red for first class, blue for second
    if colors is None:
        colors = ["red", "blue"]

    classes = sorted(df[color_col].dropna().unique())

    plt.figure(figsize=(8, 5))

    for i, c in enumerate(classes):
        data = df[df[color_col] == c][col].dropna()
        color = colors[i % len(colors)]  # cycle if more than 2 classes

        # Histogram
        plt.hist(
            data,
            bins=bins,
            density=density,
            alpha=0.4,
            label=labels.get(c, c),
            color=color,
        )

        # KDE line
        if kde and len(data) > 1:
            kde_est = gaussian_kde(data)
            x_vals = np.linspace(data.min(), data.max(), 300)
            plt.plot(x_vals, kde_est(x_vals), color=color)

    plt.xlabel(col)
    plt.ylabel("Density" if density else "Count")
    plt.title(f"{col} Distribution by {color_col}")
    plt.legend()
    plt.show()
