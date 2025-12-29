import math
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st


def doane_bins(data):
    """
    Calculates the optimal number of bins for a histogram using Doane's formula.
    Reference: https://en.wikipedia.org/wiki/Histogram#Doane's_formula

    Parameters:
        data : array-like
            Input numerical data.

    Returns:
        int : optimal number of bins
    """
    N = len(data)
    skewness = st.skew(data)
    sigma_g1 = math.sqrt((6 * (N - 2)) / ((N + 1) * (N + 3)))
    num_bins = 1 + math.log(N, 2) + math.log(1 + abs(skewness) / sigma_g1, 2)
    return round(num_bins)


def fit_distributions(data):
    """
    Fits a set of continuous distributions to the data and calculates SSE with histogram.

    Parameters:
        data : array-like
            Input numerical data.

    Returns:
        dict : mapping distribution -> [sse, shape_params, loc, scale], sorted by SSE
    """
    # List of candidate distributions
    MY_DISTRIBUTIONS = [
        st.beta,
        st.expon,
        st.norm,
        st.uniform,
        st.johnsonsb,
        st.gennorm,
        st.gausshyper,
    ]

    # Histogram for SSE calculation
    num_bins = doane_bins(data)
    frequencies, bin_edges = np.histogram(data, num_bins, density=True)
    central_values = [
        (bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)
    ]

    results = {}
    for distribution in MY_DISTRIBUTIONS:
        # Suppress runtime warnings during fitting (common for beta/johnsonsb)
        with np.errstate(divide="ignore", invalid="ignore"):
            params = distribution.fit(data)

        # Unpack parameters
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        # Evaluate PDF at bin centers
        pdf_values = np.array(
            [distribution.pdf(c, *arg, loc=loc, scale=scale) for c in central_values]
        )

        # Replace nan/inf with zero
        pdf_values = np.nan_to_num(pdf_values, nan=0.0, posinf=0.0)

        # Sum of squared errors (histogram density vs PDF)
        sse = np.sum((frequencies - pdf_values) ** 2)

        # Store results
        results[distribution] = [sse, arg, loc, scale]

    # Sort distributions by SSE (best fit first)
    results = {k: results[k] for k in sorted(results, key=results.get)}
    return results


def plot_fitted_histogram(data, results, n, limit_x_outliers=False, clip_y=False):
    """
    Plots a histogram of the data along with PDFs of top N fitted distributions.

    Parameters:
        data : array-like
            Input data to plot
        results : dict
            Fitting results from fit_distributions()
        n : int
            Number of top distributions to plot
        limit_x_outliers : bool
            If True, limit x-range to 0.5–99.5 percentiles to avoid spikes
        clip_y : bool
            If True, clip extreme PDF values to 99th percentile for plotting
    """
    # Select top N distributions by SSE
    top_distributions = {k: results[k] for k in list(results)[:n]}

    # Plot histogram
    plt.figure(figsize=(10, 5))
    plt.hist(data, density=True, ec="white", color=(63 / 235, 149 / 235, 170 / 235))
    plt.title("Histogram")
    plt.xlabel("Values")
    plt.ylabel("Frequencies")

    # Plot fitted distributions
    for distribution, result in top_distributions.items():
        sse = result[0]
        arg = result[1]
        loc = result[2]
        scale = result[3]

        # Get theoretical support of the distribution
        x_min, x_max = distribution.support(*arg, loc=loc, scale=scale)

        # Optionally limit x to robust data percentiles to avoid spikes
        if limit_x_outliers:
            p_low, p_high = np.percentile(data, [0.5, 99.5])
        else:
            p_low, p_high = (min(data), max(data))

        # Intersection of support and data range
        plot_min = max(p_low, x_min)
        plot_max = min(p_high, x_max)

        if plot_min >= plot_max:
            continue  # skip if no valid range

        # X values for PDF plotting
        x_plot = np.linspace(plot_min, plot_max, 1000)

        try:
            y_plot = distribution.pdf(x_plot, *arg, loc=loc, scale=scale)
            # Replace NaN/inf with zero
            y_plot = np.nan_to_num(y_plot, nan=0.0, posinf=0.0)
            # Optionally clip extreme values for better visualization
            if clip_y:
                y_plot = np.clip(y_plot, 0, np.percentile(y_plot, 99))
        except Exception:
            continue  # skip unstable distributions (e.g., gausshyper in rare cases)

        # Clean distribution name
        dist_name = getattr(distribution, "name", str(distribution).split()[0])

        plt.plot(
            x_plot,
            y_plot,
            label=f"{dist_name}: {str(sse)[:6]}",
            color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)),
        )

    plt.legend(title="Distributions", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.show()
