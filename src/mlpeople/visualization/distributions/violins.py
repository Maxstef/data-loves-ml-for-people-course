import plotly.express as px


def px_violin(df=None, x=None, y=None, title=None, show=True, **kwargs):
    if title is None:
        title = y if y is not None else x

    fig = px.violin(data_frame=df, x=x, y=y, title=title, **kwargs)

    if show:
        fig.show()

    return fig
