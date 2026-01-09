from scipy.stats import norm, t
import numpy as np
import matplotlib.pyplot as plt


def plot_z_test_result(
    sample_mean, null_value, population_std, sample_size, z_score=None, p_value=None
):
    """
    Plots the normal distribution for a z-test with sample and null means.

    Parameters
    ----------
    sample_mean : float
        Mean of the sample
    null_value : float
        Null hypothesis mean
    population_std : float
        Known population standard deviation
    sample_size : int
        Sample size
    z_score : float, optional
        z-statistic, used for title
    p_value : float, optional
        p-value, used for title
    """
    se = population_std / np.sqrt(sample_size)
    x = np.linspace(null_value - 4 * se, null_value + 4 * se, 500)
    pdf = norm.pdf(x, loc=null_value, scale=se)

    plt.figure(figsize=(10, 6))
    plt.plot(x, pdf, color="blue", label="Normal distribution (σ known)")
    plt.axvline(
        null_value, color="green", linestyle="-", label=f"Null mean = {null_value:.2f}"
    )
    plt.axvline(
        sample_mean,
        color="black",
        linestyle="--",
        label=f"Sample mean = {sample_mean:.2f}",
    )
    title = "One-sample z-test"
    if z_score is not None and p_value is not None:
        title += f"\n z = {z_score:.2f}, p = {p_value:.4f}"
    plt.title(title)
    plt.xlabel("Sample mean")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


def z_test(
    observed_value, null_value, population_std, sample_size, two_sided=True, plot=False
):
    """
    Performs a z-test for a single sample mean with optional plot of the sampling distribution.

    Parameters
    ----------
    observed_value : float
        Observed sample mean
    null_value : float
        Hypothesized population mean (H0)
    population_std : float
        Known population standard deviation
    sample_size : int
        Sample size
    two_sided : bool
        Whether to calculate a two-sided p-value
    plot : bool
        If True, plots the normal sampling distribution with the sample mean

    Returns
    -------
    z_score : float
        Z-test statistic
    p_value : float
        P-value for the test
    """

    # Standard error
    se = population_std / np.sqrt(sample_size)

    # Z-statistic
    z_score = (observed_value - null_value) / se

    # P-value
    if two_sided:
        p_value = 2 * (1 - norm.cdf(abs(z_score)))
    else:
        p_value = 1 - norm.cdf(z_score) if z_score > 0 else norm.cdf(z_score)

    # Optional plot using reusable function
    if plot:
        plot_z_test_result(
            sample_mean=observed_value,
            null_value=null_value,
            population_std=population_std,
            sample_size=sample_size,
            z_score=z_score,
            p_value=p_value,
        )

    return z_score, p_value


def z_test_summary(
    sample, population_mean, population_std, alpha=0.05, two_sided=True, plot=True
):
    """
    Performs a one-sample z-test and prints a readable summary, reusing z_test.

    Parameters
    ----------
    sample : array-like
        1D array of sample observations
    population_mean : float
        Known population mean (H0)
    population_std : float
        Known population standard deviation
    alpha : float
        Significance level (default 0.05)
    two_sided : bool
        Whether to perform a two-sided test (default True)
    plot : bool
        If True, plots the z-test distribution

    Returns
    -------
    z_score : float
        z-statistic
    p_value : float
        P-value of the test
    """

    sample = np.asarray(sample)
    n = len(sample)
    sample_mean = sample.mean()

    # Reuse z_test for calculation
    z_score, p_value = z_test(
        observed_value=sample_mean,
        null_value=population_mean,
        population_std=population_std,
        sample_size=n,
        two_sided=two_sided,
        plot=False,  # we'll handle plotting separately
    )

    # Print summary
    print(f"Sample size: {n}")
    print(f"Sample mean: {sample_mean:.4f}")
    print(f"Population mean: {population_mean:.4f}")
    print(f"Population std: {population_std:.4f}")
    print(f"z-statistic: {z_score:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Significance level α: {alpha}")

    if p_value < alpha:
        print(
            f"Result: Reject the null hypothesis (H0). The sample mean is significantly different from {population_mean}."
        )
    else:
        print(
            "Result: Fail to reject the null hypothesis (H0). No significant difference detected."
        )

    # Optional plot using the reusable plotting function
    if plot:
        plot_z_test_result(
            sample_mean, population_mean, population_std, n, z_score, p_value
        )

    return z_score, p_value


def plot_t_test_result(
    sample_mean, null_value, sample_std, sample_size, t_score=None, p_value=None
):
    """
    Plots the t-distribution for a one-sample t-test with sample and null means.

    Parameters
    ----------
    sample_mean : float
        Mean of the sample
    null_value : float
        Null hypothesis mean
    sample_std : float
        Standard deviation of the sample
    sample_size : int
        Sample size
    t_score : float, optional
        t-statistic, used for title
    p_value : float, optional
        p-value, used for title
    """

    se = sample_std / np.sqrt(sample_size)
    df = sample_size - 1
    x = np.linspace(null_value - 4 * se, null_value + 4 * se, 500)
    t_pdf = t.pdf((x - null_value) / se, df=df) / se

    plt.figure(figsize=(10, 6))
    plt.plot(x, t_pdf, color="red", linestyle="--", label=f"T-distribution df={df}")
    plt.axvline(
        null_value, color="green", linestyle="-", label=f"Null mean = {null_value:.2f}"
    )
    plt.axvline(
        sample_mean,
        color="black",
        linestyle="--",
        label=f"Sample mean = {sample_mean:.2f}",
    )
    title = "One-sample t-test"
    if t_score is not None and p_value is not None:
        title += f"\n t = {t_score:.2f}, p = {p_value:.4f}"
    plt.title(title)
    plt.xlabel("Sample mean")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


def t_test(
    observed_value, null_value, sample_std, sample_size, two_sided=True, plot=False
):
    """
    Performs a t-test for a single sample mean with optional plot of the t-distribution.

    Parameters
    ----------
    observed_value : float
        Observed sample mean
    null_value : float
        Hypothesized population mean (H0)
    sample_std : float
        Sample standard deviation
    sample_size : int
        Sample size
    two_sided : bool
        Whether to calculate a two-sided p-value
    plot : bool
        If True, plots the t-distribution with the sample mean

    Returns
    -------
    t_score : float
        T-test statistic
    p_value : float
        P-value for the test
    """

    # Standard error
    se = sample_std / np.sqrt(sample_size)
    df = sample_size - 1

    # t-statistic
    t_score = (observed_value - null_value) / se

    # p-value
    if two_sided:
        p_value = 2 * (1 - t.cdf(abs(t_score), df=df))
    else:
        p_value = 1 - t.cdf(t_score, df=df) if t_score > 0 else t.cdf(t_score, df=df)

    # Optional plot using the reusable function
    if plot:
        plot_t_test_result(
            sample_mean=observed_value,
            null_value=null_value,
            sample_std=sample_std,
            sample_size=sample_size,
            t_score=t_score,
            p_value=p_value,
        )

    return t_score, p_value


def t_test_for_sample(sample, null_value, two_sided=True, plot=False):
    """
    Performs a one-sample t-test for a raw sample array using the existing t_test function.

    Parameters
    ----------
    sample : array-like
        1D array of sample observations
    null_value : float
        Hypothesized population mean (H0)
    two_sided : bool
        Whether to calculate a two-sided p-value (default True)

    Returns
    -------
    t_score : float
        t-test statistic
    p_value : float
        P-value for the test
    sample_mean : float
        Mean of the sample
    sample_std : float
        Standard deviation of the sample
    """
    sample = np.asarray(sample)
    n = len(sample)
    sample_mean = sample.mean()
    sample_std = sample.std(ddof=1)

    # Call the existing t_test function
    t_score, p_value = t_test(
        observed_value=sample_mean,
        null_value=null_value,
        sample_std=sample_std,
        sample_size=n,
        two_sided=two_sided,
        plot=plot,
    )

    return t_score, p_value, sample_mean, sample_std


def t_test_summary(sample, null_value=None, alpha=0.05, two_sided=True, plot=True):
    """
    Performs a one-sample t-test on a sample and prints an interpretation, reusing t_test_for_sample.

    Parameters
    ----------
    sample : array-like
        1D array of sample observations
    null_value : float, optional
        Hypothesized population mean (H0). Defaults to mean of the sample if None.
    alpha : float
        Significance level (default 0.05)
    two_sided : bool
        Whether to perform a two-sided test (default True)
    plot : bool
        If True, plots the t-distribution with sample mean and null mean

    Returns
    -------
    t_score : float
        t-statistic
    p_value : float
        P-value of the test
    """

    sample = np.asarray(sample)
    if null_value is None:
        null_value = sample.mean()

    t_score, p_value, sample_mean, sample_std = t_test_for_sample(
        sample, null_value=null_value, two_sided=two_sided
    )

    sample_size = len(sample)

    # Print summary
    print(f"Sample size: {sample_size}")
    print(f"Sample mean: {sample_mean:.4f}")
    print(f"Sample standard deviation: {sample_std:.4f}")
    print(f"Null hypothesis mean: {null_value:.4f}")
    print(f"t-statistic: {t_score:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"Significance level α: {alpha}")

    if p_value < alpha:
        print(
            f"Result: Reject the null hypothesis (H0). The sample mean is significantly different from {null_value}."
        )
    else:
        print(
            "Result: Fail to reject the null hypothesis (H0). No significant difference detected."
        )

    # Optional plot
    if plot:
        plot_t_test_result(
            sample_mean, null_value, sample_std, sample_size, t_score, p_value
        )

    return t_score, p_value


# This function demonstrates the difference between:
# 1) Scale (standard error), which depends on σ or s
# 2) Shape (tail heaviness), which depends only on degrees of freedom
# Sample variance affects width, not tail thickness.
def plot_z_vs_t(
    sample, population_mean, population_std, title="Z vs T Distribution for Sample Mean"
):
    """
    Plots Z and T distributions for a sample mean compared to a known population mean.

    Parameters
    ----------
    sample : array-like
        The sample data
    population_mean : float
        Known population mean
    population_std : float
        Known population standard deviation (for Z-test)
    title : str
        Title for the plot
    """
    sample = np.asarray(sample)
    n = len(sample)
    df = n - 1
    sample_mean = sample.mean()
    sample_std = sample.std(ddof=1)

    # Standard errors
    se_z = population_std / np.sqrt(n)
    se_t = sample_std / np.sqrt(n)

    # X-axis range for plotting
    x_min = population_mean - 4 * max(se_z, se_t)
    x_max = population_mean + 4 * max(se_z, se_t)
    x = np.linspace(x_min, x_max, 500)

    # Z distribution (normal)
    z_pdf = norm.pdf(x, loc=population_mean, scale=se_z)

    # T distribution (sample SD)
    t_pdf = t.pdf((x - population_mean) / se_t, df=df) / se_t

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, z_pdf, label=f"Z-distribution (σ known), SE={se_z:.2f}", color="blue")
    plt.plot(
        x,
        t_pdf,
        label=f"T-distribution (σ estimated), df={df}, SE={se_t:.2f}",
        color="red",
        linestyle="--",
    )
    plt.axvline(
        sample_mean,
        color="black",
        linestyle=":",
        label=f"Sample mean = {sample_mean:.2f}",
    )
    plt.axvline(
        population_mean,
        color="green",
        linestyle="-",
        label=f"Population mean = {population_mean:.2f}",
    )
    plt.title(title)
    plt.xlabel("Sample mean")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()


# --------------------------
# Two-sample Z-test
# --------------------------
def two_sample_z_test(
    sample1, sample2, population_std1, population_std2, two_sided=True, plot=False
):
    """
    Performs a two-sample z-test (independent samples, known population stds).

    Parameters
    ----------
    sample1, sample2 : array-like
        Independent samples
    population_std1, population_std2 : float
        Known population standard deviations
    two_sided : bool
        Two-sided test if True
    plot : bool
        Whether to plot Z distribution

    Returns
    -------
    z_stat : float
        Z-test statistic
    p_value : float
        P-value
    """
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = sample1.mean(), sample2.mean()

    mean_diff = mean1 - mean2
    se = np.sqrt(population_std1**2 / n1 + population_std2**2 / n2)

    z_stat = (mean_diff) / se
    if two_sided:
        p_value = 2 * (1 - norm.cdf(abs(z_stat)))
    else:
        p_value = 1 - norm.cdf(z_stat) if z_stat > 0 else norm.cdf(z_stat)

    if plot:
        # Plot the Z-distribution of the difference
        x = np.linspace(mean_diff - 4 * se, mean_diff + 4 * se, 500)
        pdf = norm.pdf(x, loc=0, scale=se)
        plt.figure(figsize=(10, 6))
        plt.plot(x, pdf, label=f"Z-distribution (σ known), SE={se:.2f}", color="blue")
        plt.axvline(
            mean_diff,
            color="black",
            linestyle="--",
            label=f"Mean difference = {mean_diff:.2f}",
        )
        plt.axvline(0, color="green", linestyle="-", label="Null difference = 0")
        plt.title(f"Two-sample Z-test\nz = {z_stat:.2f}, p = {p_value:.4f}")
        plt.xlabel("Difference of sample means")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.show()

    return z_stat, p_value


# --------------------------
# Two-sample T-test
# --------------------------
def two_sample_t_test(sample1, sample2, equal_var=True, two_sided=True, plot=False):
    """
    Performs a two-sample t-test (independent samples, unknown population stds).
    Can perform pooled t-test (equal_var=True) or Welch's t-test (equal_var=False).

    Returns
    -------
    t_stat : float
        T-test statistic
    p_value : float
        P-value
    df : float
        Degrees of freedom
    """
    sample1 = np.asarray(sample1)
    sample2 = np.asarray(sample2)
    n1, n2 = len(sample1), len(sample2)
    mean1, mean2 = sample1.mean(), sample2.mean()
    s1, s2 = sample1.std(ddof=1), sample2.std(ddof=1)

    mean_diff = mean1 - mean2

    if equal_var:
        # Pooled variance
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / df
        se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
    else:
        # Welch's t-test
        se = np.sqrt(s1**2 / n1 + s2**2 / n2)
        df = (s1**2 / n1 + s2**2 / n2) ** 2 / (
            (s1**2 / n1) ** 2 / (n1 - 1) + (s2**2 / n2) ** 2 / (n2 - 1)
        )

    t_stat = mean_diff / se
    if two_sided:
        p_value = 2 * (1 - t.cdf(abs(t_stat), df=df))
    else:
        p_value = 1 - t.cdf(t_stat, df=df) if t_stat > 0 else t.cdf(t_stat, df=df)

    if plot:
        # Plot T-distribution of the difference
        x_min = mean_diff - 4 * se
        x_max = mean_diff + 4 * se
        x = np.linspace(x_min, x_max, 500)
        t_pdf = t.pdf((x - 0) / se, df=df) / se  # Null difference = 0
        z_pdf = norm.pdf(x, loc=0, scale=se)

        plt.figure(figsize=(10, 6))
        plt.plot(x, z_pdf, color="blue", label=f"Z approx (σ unknown, SE={se:.2f})")
        plt.plot(
            x,
            t_pdf,
            color="red",
            linestyle="--",
            label=f"T-distribution df={df:.1f}, SE={se:.2f}",
        )
        plt.axvline(
            mean_diff,
            color="black",
            linestyle=":",
            label=f"Mean difference = {mean_diff:.2f}",
        )
        plt.axvline(0, color="green", linestyle="-", label="Null difference = 0")
        plt.title(
            f"Two-sample T-test (Equal var={equal_var})\nt = {t_stat:.2f}, p = {p_value:.4f}"
        )
        plt.xlabel("Difference of sample means")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)
        plt.show()

    return t_stat, p_value, df
