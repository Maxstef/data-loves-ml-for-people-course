def lighten_color(color, amount=0.5):
    """
    Lightens the given color by blending it with white.

    Parameters:
    - color: named color or RGB tuple
    - amount: fraction of white to mix in (0 = original color, 1 = white)

    Returns:
    - RGB tuple of lightened color
    """
    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = mc.to_rgb(c)
    c = [(1 - amount) * 1 + amount * ci for ci in c]
    return c
