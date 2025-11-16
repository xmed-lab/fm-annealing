import numpy as np
from scipy import optimize
from sklearn.metrics import r2_score, mean_squared_error

epsilon = 1e-8


# ---------------------------------------------------------
# 1. Forward effect S: ∫ η(t) dt
# ---------------------------------------------------------
def compute_S(steps, lrs):
    """
    Compute the forward effect S as the integral of the learning rate
    over training steps:
        S(t_i) = ∑_{j<=i} η_j (step_j - step_{j-1})

    Parameters
    ----------
    steps : array-like, shape (T,)
        Monotonically increasing training steps.
    lrs : array-like, shape (T,)
        Learning rate at each step.

    Returns
    -------
    S : np.ndarray, shape (T,)
        Cumulative forward effect at each step.
    """
    steps = np.array(steps).flatten()
    lrs = np.array(lrs).flatten()

    if steps.ndim != 1 or lrs.ndim != 1:
        raise ValueError("steps and lrs must be 1D arrays.")
    if len(steps) != len(lrs):
        raise ValueError("steps and lrs must have the same length.")

    S = np.zeros_like(lrs, dtype=float)
    S[0] = lrs[0] * steps[0]

    for i in range(1, len(lrs)):
        S[i] = S[i - 1] + lrs[i] * (steps[i] - steps[i - 1])

    return S


# ---------------------------------------------------------
# 2. Annealing momentum S2:  Annealing Momentum (Adam-Style)
# ---------------------------------------------------------
def compute_M(
    steps,
    lrs,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
    vi_scale=1e5,
):
    """
    Compute the annealing momentum M, based on Adam-style first and second
    moments of the LR change during decay:

        delta_eta_i = (η_{i-1} - η_i) * (step_i - step_{i-1})

    We update:
        m_i = β1 m_{i-1} + (1 - β1) * delta_eta_i
        v_i = β2 v_{i-1} + (1 - β2) * delta_eta_i^2

    Then define an annealing accumulator:
        M_i = M_{i-1} + m_i / (sqrt(v_i) + ε)

    Parameters
    ----------
    steps : array-like, shape (T,)
        Monotonically increasing training steps.
    lrs : array-like, shape (T,)
        Learning rate at each step.
    beta1 : float
        First moment decay (Adam-style).
    beta2 : float
        Second moment decay (Adam-style).
    epsilon : float
        Numerical stability constant.
    vi_scale : float
        Scaling factor applied to the second moment to avoid underflow.

    Returns
    -------
    M : np.ndarray, shape (T,)
        Annealing momentum accumulator at each step.
    """
    steps = np.array(steps).flatten()
    lrs = np.array(lrs).flatten()

    if len(steps) != len(lrs):
        raise ValueError("steps and lrs must have the same length.")
    if not np.all(np.diff(steps) > 0):
        raise ValueError("steps must be strictly increasing.")

    T = len(lrs)
    m = np.zeros(T, dtype=float)   # first moment
    v = np.zeros(T, dtype=float)   # second moment
    M = np.zeros(T, dtype=float)  # cumulative annealing momentum

    for i in range(1, T):
        width = steps[i] - steps[i - 1]
        delta_eta = (lrs[i - 1] - lrs[i]) * width

        # Adam-style first and second moments
        m[i] = beta1 * m[i - 1] + (1.0 - beta1) * delta_eta
        v[i] = beta2 * v[i - 1] + (1.0 - beta2) * (delta_eta ** 2)

        # Bias correction
        m_hat = m[i] / (1.0 - beta1 ** (i + 1))
        v_hat = v[i] / (1.0 - beta2 ** (i + 1))
        v_hat = v_hat * vi_scale  # optional rescaling

        M[i] = M[i - 1] + m_hat / (np.sqrt(v_hat + epsilon) + epsilon)

    return M


# ---------------------------------------------------------
# 3. Forward-Momentum Scaling Law: L(S, N, M)
# ---------------------------------------------------------
def forward_momentum_model(params, S, N, M):
    """
    Forward-Momentum Scaling Law used in the paper:

        L = L0 + A * S^{-alpha_S} + B * N^{-alpha_N} - C * M

    where
        S : forward effect integral (S)
        N : model size (e.g., #parameters)
        M : annealing momentum integral (M)

    Parameters
    ----------
    params : iterable of 7 floats
        (L0, alpha_S, alpha_N, gamma_dummy, A, B, C).
        Note: gamma_dummy is kept for compatibility with the original code.
    S : array-like
        Forward integral S.
    N : array-like or scalar
        Model size. If scalar, it is broadcast to S's shape.
    M : array-like
        Annealing momentum M.

    Returns
    -------
    L_pred : np.ndarray
        Predicted loss values.
    """
    L0, alpha_S, alpha_N, gamma_dummy, A, B, C = params

    S = np.array(S, dtype=float)
    M = np.array(M, dtype=float)
    N = np.array(N, dtype=float)

    if N.ndim == 0:
        N = np.full_like(S, N, dtype=float)

    return L0 + A * (S ** (-alpha_S)) + B * (N ** (-alpha_N)) - C * M


# ---------------------------------------------------------
# 4. Fitting Forward-Momentum Scaling Law
# ---------------------------------------------------------
def fit_forward_momentum(
    loss,
    S,
    M,
    N,
    initial_params,
    label="forward-momentum",
    delta=1e-3,
    max_iter=1000,
    tol=1e-6,
    bounds=None,
    verbose=True,
    plot=False,
):
    """
    Fit the forward-momentum scaling law to observed loss values:

        L_obs ≈ forward_momentum_model(params, S, N, M)

    使用 Huber 损失做鲁棒拟合。

    Parameters
    ----------
    loss : array-like
        Observed (final) loss values.
    S : array-like
        Forward integral (S) for each configuration.
    M : array-like
        Annealing momentum integral (M) for each configuration.
    N : array-like or scalar
        Model size(s) corresponding to each configuration.
    initial_params : list or tuple of 7 floats
        Initial guess for (L0, alpha_S, alpha_N, gamma_dummy, A, B, C).
    label : str
        Label for printing / plotting.
    delta : float
        Huber loss threshold.
    max_iter : int
        Maximum iterations for L-BFGS-B.
    tol : float
        Convergence tolerance.
    bounds : list[tuple] or None
        Bounds for parameters. If None, reasonable defaults are used.
    verbose : bool
        Whether to print fitted parameters and metrics.
    plot : bool
        If True, scatter-plot of observed vs fitted loss is shown.

    Returns
    -------
    y_fit : np.ndarray
        Fitted loss values.
    r2 : float
        R-squared.
    mse : float
        Mean squared error.
    params_hat : tuple
        Fitted parameters (L0, alpha_S, alpha_N, gamma_dummy, A, B, C).
    """

    loss = np.array(loss, dtype=float)
    S = np.array(S, dtype=float)
    M = np.array(M, dtype=float)
    N = np.array(N, dtype=float)
    if N.ndim == 0:
        N = np.full_like(S, N, dtype=float)

    # Huber loss
    def huber(residual):
        abs_res = np.abs(residual)
        quadratic = 0.5 * residual ** 2
        linear = delta * (abs_res - 0.5 * delta)
        return np.where(abs_res <= delta, quadratic, linear)

    def objective(params):
        pred = forward_momentum_model(params, S, N, M)
        residuals = pred - loss
        return np.sum(huber(residuals))

    if bounds is None:
        # L0, alpha_S, alpha_N, gamma_dummy, A, B, C
        bounds = [
            (1.1, 8.0),   # L0
            (0.1, 10.0),  # alpha_S
            (0.1, 10.0),  # alpha_N
            (0.1, 10.0),  # gamma_dummy (unused, but kept for compatibility)
            (1e-3, 10.0), # A
            (0.0, 10.0),  # B
            (0.0, 10.0),  # C
        ]

    options = {"maxiter": max_iter, "gtol": tol, "disp": verbose}
    result = optimize.minimize(
        objective,
        np.array(initial_params, dtype=float),
        method="L-BFGS-B",
        bounds=bounds,
        options=options,
    )

    if not result.success:
        raise RuntimeError("Optimization failed: " + result.message)

    params_hat = tuple(result.x)
    y_fit = forward_momentum_model(params_hat, S, N, M)

    r2 = r2_score(loss, y_fit)
    mse = mean_squared_error(loss, y_fit)

    if verbose:
        L0, alpha_S, alpha_N, gamma_dummy, A, B, C = params_hat
        print(f"[{label}] Fitted parameters:")
        print(f"  L0       = {L0:.4f}")
        print(f"  alpha_S  = {alpha_S:.4f}")
        print(f"  alpha_N  = {alpha_N:.4f}")
        print(f"  gamma(*) = {gamma_dummy:.4f} (unused, kept for compatibility)")
        print(f"  A        = {A:.4f}")
        print(f"  B        = {B:.4f}")
        print(f"  C        = {C:.4f}")
        print(f"  R^2      = {r2:.4f}")
        print(f"  MSE      = {mse:.4e}")

    if plot:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.scatter(loss, y_fit, alpha=0.7)
        plt.plot([loss.min(), loss.max()],
                 [loss.min(), loss.max()],
                 linestyle="--")
        plt.xlabel("Observed loss")
        plt.ylabel("Fitted loss")
        plt.title(f"Forward–Momentum fit: {label}")
        plt.tight_layout()
        plt.show()

    return y_fit, r2, mse, params_hat

