import numpy as np

from synthetic_cyclic_scm import sample_exogenous_noise


def ce_linear_scm(W, treatment_idx, outcome_idx, t1=1.0, t0=0.0):
    """
    Compute the hard-intervention CE in the linear equilibrium SCM

        X = W X + U,

    where ``W[i, j]`` encodes the directed edge ``j -> i`` and the exogenous
    noise satisfies ``E[U] = 0``.

    Let ``T`` denote the treatment node, ``Y`` the outcome node, and let
    ``K = V \\ {T}`` be the indices of all non-treatment variables. Under the
    intervention ``do(T = t)``, the treatment is fixed externally and the
    remaining structural equations become

        X_K = W_KK X_K + W_KT t + U_K.

    Rearranging gives

        (I - W_KK) X_K = W_KT t + U_K,

    so the post-intervention equilibrium is

        X_K^(t) = (I - W_KK)^(-1) (W_KT t + U_K).

    Taking expectations and using ``E[U_K] = 0`` yields

        E[X_K^(t)] = (I - W_KK)^(-1) W_KT t.

    Therefore the causal effect comparing two treatment levels
    ``t1`` and ``t0`` is

        CE(t1, t0)
        = E[Y | do(T = t1)] - E[Y | do(T = t0)]
        = e_Y^T (I - W_KK)^(-1) W_KT (t1 - t0),

    where ``e_Y`` selects the outcome coordinate inside ``K``. The code below
    implements exactly this formula by solving the linear system once for the
    response vector ``(I - W_KK)^(-1) W_KT`` and then extracting the outcome
    entry.
    """
    n = W.shape[0]
    keep = np.array([i for i in range(n) if i != treatment_idx])

    W_kk = W[np.ix_(keep, keep)]
    W_kt = W[keep, treatment_idx]

    response = np.linalg.solve(np.eye(n - 1) - W_kk, W_kt)

    y_pos = np.where(keep == outcome_idx)[0][0]
    return float(response[y_pos] * (t1 - t0))


def solve_do_tanh_plus_u_batch(
    W,
    U_batch,
    treatment_idx,
    treatment_value,
    tol=1e-8,
    max_iter=1_000,
):
    """
    Batched solver for nonlinear simple SCM:
        X = tanh(WX) + U

    Under hard intervention:
        do(X_treatment = treatment_value)

    U_batch shape: (n_samples, n_nodes)
    """
    n_samples, n = U_batch.shape

    keep = np.array([i for i in range(n) if i != treatment_idx])

    W_kk = W[np.ix_(keep, keep)]
    W_kt = W[keep, treatment_idx]
    U_k = U_batch[:, keep]

    X_k = np.zeros((n_samples, len(keep)))

    for _ in range(max_iter):
        X_new = np.tanh(X_k @ W_kk.T + treatment_value * W_kt) + U_k

        if np.max(np.abs(X_new - X_k)) < tol:
            X = np.zeros((n_samples, n))
            X[:, keep] = X_new
            X[:, treatment_idx] = treatment_value
            return X

        X_k = X_new

    raise RuntimeError("Fixed-point iteration did not converge")


def ce_nonlinear_scm_fast(
    W,
    treatment_idx,
    outcome_idx,
    t1=1.0,
    t0=0.0,
    n_mc=100_000,
    sigma_range=(1.0, 1.0),
    noise_type="gaussian",
    seed=None,
    tol=1e-8,
    max_iter=1_000,
):
    """
    Fast Monte Carlo CE for nonlinear simple SCM:
        X = tanh(WX) + U

    Computes:
        E_U[Y^{do(T=t1)}(U) - Y^{do(T=t0)}(U)]
    """
    rng = np.random.default_rng(seed)
    n = W.shape[0]

    U_batch = sample_exogenous_noise(
        n_nodes=n,
        n_samples=n_mc,
        sigma_range=sigma_range,
        noise_type=noise_type,
        rng=rng,
    )

    X1 = solve_do_tanh_plus_u_batch(
        W=W,
        U_batch=U_batch,
        treatment_idx=treatment_idx,
        treatment_value=t1,
        tol=tol,
        max_iter=max_iter,
    )

    X0 = solve_do_tanh_plus_u_batch(
        W=W,
        U_batch=U_batch,
        treatment_idx=treatment_idx,
        treatment_value=t0,
        tol=tol,
        max_iter=max_iter,
    )

    return float(np.mean(X1[:, outcome_idx] - X0[:, outcome_idx]))


def solve_do_tanh_plus_u_batch(
    W,
    U_batch,
    treatment_idx,
    treatment_value,
    tol=1e-8,
    max_iter=1000,
):
    """
    SCM:
        X = tanh(WX) + U

    Intervention:
        do(X_treatment = treatment_value)

    W[i, j] means j -> i.
    U_batch shape: (n_samples, n_nodes)
    """
    n_samples, n = U_batch.shape
    keep = np.array([i for i in range(n) if i != treatment_idx])

    W_kk = W[np.ix_(keep, keep)]
    W_kt = W[keep, treatment_idx]
    U_k = U_batch[:, keep]

    X_k = np.zeros((n_samples, len(keep)))

    for _ in range(max_iter):
        X_new = np.tanh(X_k @ W_kk.T + treatment_value * W_kt) + U_k

        if np.max(np.abs(X_new - X_k)) < tol:
            X = np.zeros((n_samples, n))
            X[:, keep] = X_new
            X[:, treatment_idx] = treatment_value
            return X

        X_k = X_new

    raise RuntimeError("Fixed-point iteration did not converge")


def ce_continuous_treatment_tanh_plus_u(
    W,
    treatment_idx,
    outcome_idx,
    t1,
    t0,
    n_mc=10_000,
    noise_scale=1.0,
    seed=None,
    tol=1e-8,
    max_iter=1000,
):
    """
    Continuous-treatment causal contrast:
        E[Y^{do(T=t1)} - Y^{do(T=t0)}]
    """
    rng = np.random.default_rng(seed)
    n = W.shape[0]

    U_batch = rng.normal(0.0, noise_scale, size=(n_mc, n))

    X1 = solve_do_tanh_plus_u_batch(
        W, U_batch, treatment_idx, t1, tol=tol, max_iter=max_iter
    )

    X0 = solve_do_tanh_plus_u_batch(
        W, U_batch, treatment_idx, t0, tol=tol, max_iter=max_iter
    )

    return float(np.mean(X1[:, outcome_idx] - X0[:, outcome_idx]))


# Example usage
if __name__ == "__main__":
    W = np.array(
        [
            [0.0, 0.2, 0.0],
            [0.1, 0.0, 0.3],
            [0.0, 0.4, 0.0],
        ]
    )

    treatment_idx = 0
    outcome_idx = 2

    print("Linear CE:")
    print(ce_linear_scm(W, treatment_idx, outcome_idx))

    print("Fast nonlinear CE:")
    print(
        ce_nonlinear_scm_fast(
            W,
            treatment_idx,
            outcome_idx,
            n_mc=100_000,
            sigma_range=(1.0, 1.0),
            noise_type="gaussian",
            seed=123,
        )
    )
