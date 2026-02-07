import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt


def px_violin(df=None, x=None, y=None, title=None, show=True, **kwargs):
    if title is None:
        title = y if y is not None else x

    fig = px.violin(data_frame=df, x=x, y=y, title=title, **kwargs)

    if show:
        fig.show()

    return fig


def plt_violin(df, y, x=None, title=None, colors=None):
    """
    Draw a violin plot using hue for coloring (future-proof).

    Parameters:
    df     : pandas DataFrame
    y      : str, numeric column to plot
    x      : str, categorical column for x-axis (optional)
    title  : str, plot title (optional)
    colors : list of colors to use for classes (optional)
    """
    plt.figure(figsize=(8, 5))

    df = df.copy()  # temporary copy just for plotting

    if x:
        # Convert x to string to ensure consistent hue mapping
        df["_x_str"] = df[x].astype(str)
        unique_vals = sorted(df["_x_str"].unique())

        # Default colors if not provided
        if colors is None:
            # cycle through some standard colors if >2 classes
            colors = ["red", "blue", "green", "orange", "purple", "brown"]

        # Build dynamic palette based on unique x values
        palette = {
            str(val): colors[i % len(colors)] for i, val in enumerate(unique_vals)
        }

        # Plot using hue
        sns.violinplot(
            x=x,  # Keep original x for positioning
            y=y,
            data=df,
            hue="_x_str",
            palette=palette,
            inner="quartile",
            dodge=True,
            legend=False,  # prevent duplicate legends
        )
        plt.xlabel(x)
        plt.ylabel(y)
        if not title:
            title = f"{y} Distribution by {x}"

        # clean up temporary column
        df.drop(columns=["_x_str"], inplace=True)

    else:
        # Single-column violin plot
        sns.violinplot(y=y, data=df, color="lightblue", inner="quartile")
        plt.ylabel(y)
        if not title:
            title = f"{y} Distribution"

    plt.title(title)
    plt.show()
