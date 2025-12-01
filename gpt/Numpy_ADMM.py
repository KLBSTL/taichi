import numpy as np
from numpy.linalg import cholesky, solve, norm

def soft_threshold(v, kappa):
    return np.sign(v) * np.maximum(np.abs(v) - kappa, 0.0)

def admm_lasso(A, b, lam, rho=1.0, alpha=1.0, max_iters=1000,
               abs_tol=1e-4, rel_tol=1e-3, verbose=False):
    """
    Solve min_x 0.5||A x - b||^2 + lam * ||x||_1 using ADMM
    (x,z) split with x-update via linear solve and z via soft-threshold.
    """
    m, n = A.shape
    AtA = A.T @ A
    Atb = A.T @ b

    # Pre-factorize (AtA + rho I) via Cholesky (must be SPD)
    L = cholesky(AtA + rho * np.eye(n))
    L_T = L.T

    # initial
    x = np.zeros(n)
    z = np.zeros(n)
    u = np.zeros(n)

    history = {'objval': [], 'r_norm': [], 's_norm': [], 'eps_pri': [], 'eps_dual': []}

    for k in range(max_iters):
        # x-update: solve (AtA + rho I) x = Atb + rho (z - u)
        q = Atb + rho * (z - u)  # rhs
        # solve L y = q, then L_T x = y
        y = solve(L, q)
        x = solve(L_T, y)

        # over-relaxation
        x_hat = alpha * x + (1 - alpha) * z

        # z-update: soft-thresholding
        z_old = z.copy()
        z = soft_threshold(x_hat + u, lam / rho)

        # u-update (scaled dual)
        u += x_hat - z

        # diagnostics, residuals, objective
        r = x - z
        s = -rho * (z - z_old)

        r_norm = norm(r)
        s_norm = norm(s)

        eps_pri = np.sqrt(n) * abs_tol + rel_tol * max(norm(x), norm(z))
        eps_dual = np.sqrt(n) * abs_tol + rel_tol * norm(rho * u)

        obj = 0.5 * norm(A @ x - b)**2 + lam * np.sum(np.abs(z))  # use z for sparsity
        history['objval'].append(obj)
        history['r_norm'].append(r_norm)
        history['s_norm'].append(s_norm)
        history['eps_pri'].append(eps_pri)
        history['eps_dual'].append(eps_dual)

        if verbose and (k % 50 == 0 or k == max_iters - 1):
            print(f"iter {k}: obj {obj:.4e}, r {r_norm:.4e}, s {s_norm:.4e}, eps_pri {eps_pri:.2e}, eps_dual {eps_dual:.2e}")

        if r_norm <= eps_pri and s_norm <= eps_dual:
            if verbose:
                print(f"Converged in {k} iters.")
            break

        # optional adaptive rho
        if r_norm > 10 * s_norm:
            rho *= 2
            u /= 2
            # refactor L for new rho
            L = cholesky(AtA + rho * np.eye(n))
            L_T = L.T
        elif s_norm > 10 * r_norm:
            rho /= 2
            u *= 2
            L = cholesky(AtA + rho * np.eye(n))
            L_T = L.T

    return x, history

# ---- usage example ----
if __name__ == "__main__":
    np.random.seed(0)
    m, n = 150, 300
    A = np.random.randn(m, n)
    x_true = np.zeros(n); x_true[np.random.choice(n, 10, replace=False)] = np.random.randn(10)
    b = A @ x_true + 0.5 * np.random.randn(m)
    lam = 0.1

    x_hat, hist = admm_lasso(A, b, lam, rho=1.0, alpha=1.0, max_iters=2000, verbose=True)
    print("reconstruction error:", np.linalg.norm(x_hat - x_true))
