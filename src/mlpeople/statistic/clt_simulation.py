import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


def clt_sample_proportions(
    experiment_count=1000,
    sample_size=100,
    outcome_count=2,
    bins=None,
    overlay_normal=True,
    random_seed=None,
):
    """
    Simulates the Central Limit Theorem for the proportion of a specific outcome (e.g., '1')
    in repeated samples drawn from a discrete uniform population (e.g., coin flips or dice).

    Each sample contains 'sample_size' trials, repeated 'experiment_count' times.
    Plots a histogram of the sample proportions and optionally overlays the theoretical normal distribution.

    Parameters:
    - experiment_count: Number of repeated samples (experiments)
    - sample_size: Number of trials per sample
    - outcome_count: Number of possible outcomes per trial (e.g., 2 for coin, 6 for dice)
    - bins: Number of histogram bins. Automatically calculated if None.
    - overlay_normal: If True, overlays the theoretical normal distribution of sample proportions

    Returns:
    - sample_proportions: Array of sample proportions
    - population_mean: Theoretical probability of outcome '1'
    - sample_variance: Theoretical variance of sample proportions
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    if bins is None:
        bins = min(30, int(np.sqrt(experiment_count)))

    # Generate all trials
    samples = np.random.randint(
        1, outcome_count + 1, size=(experiment_count, sample_size)
    )

    # Calculate proportion of "1"s in each sample
    sample_proportions = (samples == 1).mean(axis=1)

    # Theoretical mean and variance of proportion
    p = 1 / outcome_count
    population_mean = p
    population_variance = p * (1 - p)
    sample_variance = population_variance / sample_size
    sample_std = np.sqrt(sample_variance)

    print(f"Population mean (theory): {population_mean}")
    print(f"Average of sample proportions: {np.mean(sample_proportions):.4f}\n")

    print(f"Population variance (theory): {population_variance:.4f}")
    print(f"Theoretical variance of sample proportion (σ²/N): {sample_variance:.4f}")
    print(f"Variance of sample proportions: {np.var(sample_proportions):.4f}")

    # Plot histogram
    plt.hist(
        sample_proportions,
        bins=bins,
        density=True,
        alpha=0.6,
        edgecolor="black",
        label="Sample proportions",
    )

    # Overlay theoretical normal distribution
    if overlay_normal:
        x = np.linspace(min(sample_proportions), max(sample_proportions), 1000)
        plt.plot(
            x,
            norm.pdf(x, loc=population_mean, scale=sample_std),
            "r-",
            lw=2,
            label="Theoretical normal PDF",
        )

    plt.xlabel("Proportion of ones")
    plt.ylabel("Density")
    plt.title(
        f"CLT Simulation: {experiment_count} experiments, {sample_size} trials per experiment"
    )
    plt.legend()
    plt.show()

    return sample_proportions, population_mean, sample_variance


def clt_sample_means(
    experiment_count=None,
    sample_size=None,
    outcome_count=6,
    bins=None,
    overlay_normal=True,
    random_seed=None,
):
    """
    Simulates the Central Limit Theorem by computing the mean of repeated samples
    drawn from a discrete uniform population (e.g., dice rolls).

    Each sample contains 'sample_size' trials, repeated 'experiment_count' times.
    Plots a histogram of the sample means and optionally overlays the theoretical normal distribution.

    Parameters:
    - experiment_count: Number of repeated samples (experiments). Defaults to 2000.
    - sample_size: Number of trials per sample. Defaults to 100.
    - outcome_count: Number of possible outcomes (e.g., 6 for a standard die).
    - bins: Number of histogram bins. Automatically calculated if None.
    - overlay_normal: If True, overlays the theoretical normal distribution of sample means.

    Returns:
    - sample_means: Array of sample means
    - population_mean: Theoretical population mean
    - population_variance: Theoretical population variance
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    # Set dynamic defaults
    if sample_size is None:
        sample_size = 100
    if experiment_count is None:
        experiment_count = 2000
    if bins is None:
        bins = min(30, int(np.sqrt(experiment_count)))

    # Generate all samples
    samples = np.random.randint(
        1, outcome_count + 1, size=(experiment_count, sample_size)
    )

    # Compute the mean of each sample
    sample_means = samples.mean(axis=1)

    # Theoretical population statistics
    population_mean = np.mean(range(1, outcome_count + 1))
    population_variance = np.var(range(1, outcome_count + 1))
    sample_mean_variance = population_variance / sample_size
    sample_mean_std = np.sqrt(sample_mean_variance)

    # Print statistics
    print(f"Population mean (theory): {population_mean}")
    print(f"Average of sample means: {np.mean(sample_means):.4f}\n")

    print(f"Population variance (theory): {population_variance:.4f}")
    print(f"Theoretical variance of sample mean (σ²/N): {sample_mean_variance:.4f}")
    print(f"Variance of sample means: {np.var(sample_means):.4f}")

    # Plot histogram
    plt.hist(
        sample_means,
        bins=bins,
        density=True,
        edgecolor="black",
        alpha=0.6,
        label="Sample means",
    )

    if overlay_normal:
        # Overlay theoretical normal distribution
        x = np.linspace(min(sample_means), max(sample_means), 1000)
        plt.plot(
            x,
            norm.pdf(x, loc=population_mean, scale=sample_mean_std),
            "r-",
            lw=2,
            label="Theoretical normal PDF",
        )

    plt.xlabel("Sample mean")
    plt.ylabel("Density")
    plt.title(
        f"Central Limit Theorem Simulation\n{experiment_count} experiments, sample size = {sample_size}"
    )
    plt.legend()
    plt.show()

    return sample_means, population_mean, population_variance


def clt_from_normal_population(
    loc=100,
    scale=15,
    size=2000,
    sample_size=25,
    experiment_count=1000,
    bins=30,
    overlay_normal=True,
    random_seed=None,
):
    """
    Simulates the Central Limit Theorem using a normally distributed population.

    A population is generated from N(loc, scale^2). Repeated samples of size
    `sample_size` are drawn (with replacement), and their means are computed.
    The distribution of sample means is compared to the theoretical normal
    distribution predicted by the CLT.

    Parameters
    ----------
    loc : float
        Mean (μ) of the normal population
    scale : float
        Standard deviation (σ) of the normal population
    size : int
        Size of the population
    sample_size : int
        Number of observations per sample (n)
    experiment_count : int
        Number of repeated samples
    bins : int
        Number of histogram bins
    overlay_normal : bool
        Whether to overlay the theoretical normal distribution
    random_seed : int or None
        Optional seed for reproducibility

    Returns
    -------
    population : np.ndarray
        Generated population data
    sample_means : np.ndarray
        Means of each sample
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    # --- Generate population ---
    population = np.random.normal(loc=loc, scale=scale, size=size)

    population_mean = population.mean()
    population_variance = population.var()
    population_std = population.std()

    print("Population statistics")
    print(f"  Mean:     {population_mean:.4f}")
    print(f"  Variance: {population_variance:.4f}")
    print(f"  Std dev:  {population_std:.4f}")

    # --- Draw samples and compute means ---
    samples = np.random.choice(
        population,
        size=(experiment_count, sample_size),
        replace=True,  # i.i.d. sampling (CLT assumption)
    )

    sample_means = samples.mean(axis=1)

    # --- Theoretical CLT values ---
    theoretical_variance = population_variance / sample_size
    theoretical_se = population_std / np.sqrt(sample_size)

    print("\nSample mean statistics")
    print(f"  Average sample mean: {sample_means.mean():.4f}")
    print(f"  Theoretical variance (σ²/n): {theoretical_variance:.4f}")
    print(f"  Observed variance:          {sample_means.var():.4f}")
    print(f"  Theoretical SE (σ/√n):      {theoretical_se:.4f}")
    print(f"  Observed SE:               {sample_means.std():.4f}")

    # --- Plot population ---
    plt.hist(population, bins=bins, density=True, alpha=0.6, edgecolor="black")
    plt.title("Population Distribution")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()

    # --- Plot sample means ---
    plt.hist(
        sample_means,
        bins=bins,
        density=True,
        alpha=0.6,
        edgecolor="black",
        label="Sample means",
    )

    if overlay_normal:
        x = np.linspace(sample_means.min(), sample_means.max(), 1000)
        plt.plot(
            x,
            norm.pdf(x, loc=population_mean, scale=theoretical_se),
            "r-",
            lw=2,
            label="Theoretical normal PDF",
        )

    plt.title(
        f"CLT from Normal Population\n"
        f"{experiment_count} samples, sample size = {sample_size}"
    )
    plt.xlabel("Sample mean")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    return population, sample_means


def clt_from_categorical_population(
    values,
    probabilities,
    target_value,
    population_size=5000,
    sample_size=50,
    experiment_count=1000,
    bins=30,
    overlay_normal=True,
    random_seed=None,
):
    """
    Simulates the Central Limit Theorem for sample proportions of a target category
    drawn from a categorical population.

    Parameters
    ----------
    values : array-like
        Possible categorical values (e.g. [1, 2, 3, 4, 5, 6])
    probabilities : array-like
        Probabilities for each value (must sum to 1)
    target_value : scalar
        Category whose sample proportion is analyzed
    population_size : int
        Size of the generated population
    sample_size : int
        Number of observations per sample
    experiment_count : int
        Number of repeated samples
    bins : int
        Number of histogram bins
    overlay_normal : bool
        Whether to overlay the theoretical normal distribution
    random_seed : int or None
        Optional random seed

    Returns
    -------
    population : np.ndarray
        Generated population
    sample_proportions : np.ndarray
        Sample proportions of target_value
    """

    if random_seed is not None:
        np.random.seed(random_seed)

    values = np.asarray(values)
    probabilities = np.asarray(probabilities)

    if not np.isclose(probabilities.sum(), 1):
        raise ValueError("Probabilities must sum to 1.")

    if target_value not in values:
        raise ValueError("target_value must be one of the values.")

    # --- Generate categorical population ---
    population = np.random.choice(values, size=population_size, p=probabilities)

    # True probability of target category
    p_target = probabilities[values == target_value][0]

    print("Population statistics")
    print(f"  Target value: {target_value}")
    print(f"  True probability p: {p_target:.4f}")

    # --- Draw samples ---
    samples = np.random.choice(
        population, size=(experiment_count, sample_size), replace=True
    )

    # Indicator: X == target_value
    sample_proportions = (samples == target_value).mean(axis=1)

    # --- Theoretical CLT values ---
    theoretical_variance = p_target * (1 - p_target) / sample_size
    theoretical_se = np.sqrt(theoretical_variance)

    print("\nSample proportion statistics")
    print(f"  Average sample proportion: {sample_proportions.mean():.4f}")
    print(f"  Theoretical variance:      {theoretical_variance:.6f}")
    print(f"  Observed variance:         {sample_proportions.var():.6f}")
    print(f"  Theoretical SE:            {theoretical_se:.6f}")
    print(f"  Observed SE:               {sample_proportions.std():.6f}")

    # --- Plot population ---
    plt.hist(population, bins=len(values), density=True, edgecolor="black", alpha=0.6)
    plt.title("Categorical Population")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.show()

    # --- Plot sample proportions ---
    plt.hist(
        sample_proportions,
        bins=bins,
        density=True,
        alpha=0.6,
        edgecolor="black",
        label="Sample proportions",
    )

    if overlay_normal:
        x = np.linspace(sample_proportions.min(), sample_proportions.max(), 1000)
        plt.plot(
            x,
            norm.pdf(x, loc=p_target, scale=theoretical_se),
            "r-",
            lw=2,
            label="Theoretical normal PDF",
        )

    plt.title(
        f"CLT for Sample Proportions\n" f"Target = {target_value}, n = {sample_size}"
    )
    plt.xlabel("Sample proportion")
    plt.ylabel("Density")
    plt.legend()
    plt.show()

    return population, sample_proportions


def clt_from_bernoulli_population(
    p=0.5,
    population_size=5000,
    sample_size=50,
    experiment_count=1000,
    bins=30,
    overlay_normal=True,
    random_seed=None,
):
    """
    Special case of clt_from_categorical_population for Bernoulli trials.

    Parameters
    ----------
    p : float
        Probability of success (P(X=1))
    population_size : int
        Size of the finite population
    sample_size : int
        Number of trials per sample
    experiment_count : int
        Number of repeated samples
    bins : int
        Number of histogram bins
    overlay_normal : bool
        Whether to overlay the theoretical normal distribution
    random_seed : int or None
        Optional random seed for reproducibility
    """

    if not (0 <= p <= 1):
        raise ValueError("p must be between 0 and 1.")

    return clt_from_categorical_population(
        values=[0, 1],
        probabilities=[1 - p, p],
        target_value=1,
        population_size=population_size,
        sample_size=sample_size,
        experiment_count=experiment_count,
        bins=bins,
        overlay_normal=overlay_normal,
        random_seed=random_seed,
    )
