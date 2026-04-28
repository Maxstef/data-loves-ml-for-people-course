import numpy as np
import matplotlib.pyplot as plt


# ---- distances ----
def pairwise_squared_distances(X):
    """
    Compute the full matrix of squared Euclidean distances between rows of X.

    Uses the identity:
        ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a·b

    Parameters
    ----------
    X : ndarray of shape (N, d)
        N points in d-dimensional space.

    Returns
    -------
    D : ndarray of shape (N, N)
        Squared distance matrix where D[i, j] = ||x_i - x_j||^2.
    """
    # Use vectorized identity to avoid explicit loops
    sum_X = np.sum(np.square(X), axis=1)
    D = sum_X[:, None] + sum_X[None, :] - 2 * X.dot(X.T)
    np.fill_diagonal(D, 0.0)
    return D


# ---- high-dim probabilities ----
def compute_entropy(P_row):
    """
    Compute Shannon entropy (base 2) of a probability distribution.

    Low entropy → one dominant neighbor.
    High entropy → many equally likely neighbors.
    """
    P_row = P_row[P_row > 1e-12]
    return -np.sum(P_row * np.log2(P_row))


def perplexity(P_row):
    """
    Compute perplexity of a probability distribution.

    Perplexity = 2 ^ entropy.

    Interpreted as the effective number of neighbors.
    """
    return 2 ** compute_entropy(P_row)


def binary_search_sigma(D_i, target_perplexity=30.0, tol=1e-5, max_iter=50):
    """
    Find sigma for a single point so that its neighborhood
    distribution has the desired perplexity.

    Parameters
    ----------
    D_i : ndarray of shape (N,)
        Squared distances from point i to all other points.
    target_perplexity : float
        Desired effective neighborhood size.
    tol : float
        Tolerance for perplexity match.
    max_iter : int
        Maximum binary search iterations.

    Returns
    -------
    sigma : float
        Optimal bandwidth for point i.
    P_i : ndarray
        Conditional probabilities p(j | i).
    """
    # Search sigma range: small → sharp distribution, large → flat
    sigma_min, sigma_max = 1e-5, 100.0

    for _ in range(max_iter):
        sigma = (sigma_min + sigma_max) / 2

        # Build Gaussian neighborhood probabilities
        P_i = np.exp(-D_i / (2 * sigma**2))
        P_i[P_i < 1e-12] = 0.0
        P_i = P_i / (np.sum(P_i) + 1e-12)

        perp = perplexity(P_i)

        # Adjust sigma based on how flat/sharp distribution is
        if perp > target_perplexity:
            sigma_max = sigma
        else:
            sigma_min = sigma

        if abs(perp - target_perplexity) < tol:
            break

    return sigma, P_i


def compute_P_perplexity(D, target_perplexity=30.0):
    """
    Compute conditional probability matrix P[i, j] = p(j | i)
    using per-point sigmas determined by perplexity.

    Returns
    -------
    P : ndarray (N, N)
        Conditional probabilities.
    sigmas : ndarray (N,)
        Sigma value for each point.
    """
    N = D.shape[0]
    P = np.zeros((N, N))
    sigmas = np.zeros(N)

    # Each point gets its own sigma_i
    for i in range(N):
        sigma_i, P_i = binary_search_sigma(D[i], target_perplexity)
        P[i] = P_i
        sigmas[i] = sigma_i

    np.fill_diagonal(P, 0.0)
    return P, sigmas


def symmetrize_P(P):
    """
    Convert conditional probabilities into symmetric joint probabilities.

        P_ij = (p_{j|i} + p_{i|j}) / sum_all

    Returns
    -------
    P_sym : ndarray (N, N)
        Symmetric joint probability matrix.
    """
    # If either i considers j close OR j considers i close → strengthen link
    P_sym = P + P.T
    P_sym /= np.sum(P_sym)
    return P_sym


# ---- low-dim similarities ----
def student_t_Q(Y):
    """
    Compute low-dimensional similarities using Student-t kernel.

        Q_ij ∝ 1 / (1 + ||y_i - y_j||^2)

    Returns
    -------
    Q : normalized similarities
    Q_num : unnormalized kernel values (used in gradient)
    """
    # Heavy-tailed kernel prevents crowding
    D = pairwise_squared_distances(Y)
    Q_num = 1.0 / (1.0 + D)
    np.fill_diagonal(Q_num, 0.0)
    Q = Q_num / (np.sum(Q_num) + 1e-12)
    return Q, Q_num


# ---- objective ----
def kl_divergence(P, Q):
    """
    Compute KL divergence KL(P || Q).

    Penalizes when close neighbors in high-dim space
    are not close in the embedding.
    """
    return np.sum(P * np.log((P + 1e-12) / (Q + 1e-12)))


# ---- main algorithm ----
def tsne_from_scratch(
    X,
    sigma=1.0,  # deprecated
    dim=2,
    lr=200.0,
    n_iter=500,
    seed=42,
    target_perplexity=30.0,
    initial_momentum=None,
    final_momentum=None,
    verbose=True,
    verbose_on_iter=100,
    show_pq_matrix_training=False,
):
    """
    Run t-SNE optimization from scratch.

    Parameters
    ----------
    X : ndarray (N, d)
        Input data.
    dim : int
        Embedding dimension.
    lr : float
        Learning rate.
    n_iter : int
        Number of optimization iterations.
    target_perplexity : float
        Controls effective neighborhood size.
    initial_momentum, final_momentum : float or None
        Momentum schedule.
    """

    rng = np.random.default_rng(seed)
    N = X.shape[0]

    # Step 1 — compute target similarities in high-dimensional space
    D_high = pairwise_squared_distances(X)
    P_cond, sigmas = compute_P_perplexity(D_high, target_perplexity)
    P = symmetrize_P(P_cond)

    # Step 2 — initialize embedding randomly near zero
    Y = 1e-4 * rng.standard_normal((N, dim))

    # Structures for momentum + adaptive learning
    Y_inc = np.zeros_like(Y)
    grads_memory = np.zeros_like(Y)
    gains = np.ones_like(Y)
    eta = lr

    if final_momentum is None and initial_momentum is not None:
        final_momentum = initial_momentum

    # for logging and visualizations
    if show_pq_matrix_training:
        Q_init, _ = student_t_Q(Y)
        max_error_ref = np.max(np.abs(P - Q_init))

    # Step 3 — optimize embedding to match P with Q
    for it in range(n_iter):
        Q, Q_num = student_t_Q(Y)

        # ---- compute gradient from pairwise forces ----
        grads = np.zeros_like(Y)
        for i in range(N):
            diff = Y[i] - Y
            coeff = (P[i] - Q[i]) * Q_num[i]
            pairwise_forces = coeff[:, None] * diff
            grads[i] = 4.0 * np.sum(pairwise_forces, axis=0)

        # ---- update embedding ----
        if initial_momentum is None:
            # Simple gradient descent
            Y -= lr * grads
        else:
            # Adaptive gains + momentum (stable training)
            prev_grads = grads_memory
            sign_change = np.sign(grads) != np.sign(prev_grads)
            grads_memory = grads

            gains = (gains + 0.2) * sign_change + (gains * 0.8) * (~sign_change)
            gains = np.clip(gains, 0.01, 100)

            momentum = initial_momentum if it < 250 else final_momentum
            Y_inc = momentum * Y_inc + eta * (gains * grads)
            Y -= Y_inc

        # ---- optional monitoring ----
        if verbose and (it + 1) % verbose_on_iter == 0:
            loss = kl_divergence(P, Q)
            print(f"iter {it+1:4d} | KL={loss:.6f}")

            if show_pq_matrix_training:
                vmax = np.percentile(np.concatenate([P.flatten(), Q.flatten()]), 99)

                plt.figure(figsize=(8, 2))
                plt.subplot(1, 3, 1)
                plt.title("P")
                plt.imshow(P, cmap="viridis", vmin=0, vmax=vmax)
                plt.colorbar()

                plt.subplot(1, 3, 2)
                plt.title("Q")
                plt.imshow(Q, cmap="viridis", vmin=0, vmax=vmax)
                plt.colorbar()

                plt.subplot(1, 3, 3)
                plt.title("|P - Q|")
                plt.imshow(np.abs(P - Q), cmap="hot", vmin=0, vmax=max_error_ref)
                plt.colorbar()

                plt.tight_layout()
                plt.show()

    return Y, P
