import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm


def load_ratings(path_csv: str):
    """
    Expects columns: user_id, movie_id, rating   where rating in {+1, -1}.
    Returns:
      u_idx (n_obs,), v_idx (n_obs,), r (n_obs,), n_users, n_movies,
      plus the ID maps (for debugging).
    """
    df = pd.read_csv(path_csv)

    required = {"user_id", "movie_id", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path_csv}: {missing}")

    # Force rating to int and validate
    r = df["rating"].astype(int).to_numpy()
    if not np.all(np.isin(r, [-1, 1])):
        bad = np.unique(r[~np.isin(r, [-1, 1])])
        raise ValueError(f"Found ratings outside {{-1,+1}}: {bad}")

    # Map user/movie IDs to contiguous indices (safe even if already 0..n-1)
    user_ids = df["user_id"].to_numpy()
    movie_ids = df["movie_id"].to_numpy()

    uniq_users, u_idx = np.unique(user_ids, return_inverse=True)
    uniq_movies, v_idx = np.unique(movie_ids, return_inverse=True)

    n_users = uniq_users.size
    n_movies = uniq_movies.size

    return u_idx, v_idx, r, n_users, n_movies, uniq_users, uniq_movies


def build_index_lists(u_idx, v_idx, n_users, n_movies):
    """
    Precompute observation indices per user and per movie.
    """
    obs_by_user = [[] for _ in range(n_users)]
    obs_by_movie = [[] for _ in range(n_movies)]
    for k, (i, j) in enumerate(zip(u_idx, v_idx)):
        obs_by_user[i].append(k)
        obs_by_movie[j].append(k)
    # Convert to numpy arrays for faster indexing
    obs_by_user = [np.asarray(lst, dtype=np.int64) for lst in obs_by_user]
    obs_by_movie = [np.asarray(lst, dtype=np.int64) for lst in obs_by_movie]
    return obs_by_user, obs_by_movie


def em_probit_mf_part_a(
    ratings_csv: str,
    d: int = 5,
    c: float = 1.0,
    sigma2: float = 1.0,
    n_iter: int = 100,
    seed: int = 0,
):
    """
    Probit matrix factorization via EM with truncated-normal latent variable:
      phi_ij ~ N(u_i^T v_j, sigma2)
      r_ij = sign(phi_ij), where r in {+1, -1}
    Priors:
      u_i ~ N(0, (sigma2/c) I),  v_j ~ N(0, (sigma2/c) I)

    Returns:
      ll: array of length n_iter with log p(R,U,V) (up to an additive constant)
    """
    u_idx, v_idx, r, n_users, n_movies, _, _ = load_ratings(ratings_csv)
    obs_by_user, obs_by_movie = build_index_lists(u_idx, v_idx, n_users, n_movies)

    rng = np.random.default_rng(seed)

    # Required initialization: Normal(0, 0.1 I)
    U = rng.normal(loc=0.0, scale=np.sqrt(0.1), size=(n_users, d))
    V = rng.normal(loc=0.0, scale=np.sqrt(0.1), size=(n_movies, d))

    ll = np.zeros(n_iter, dtype=float)

    # Constants
    I_d = np.eye(d)
    inv_sigma2 = 1.0 / sigma2

    for t in range(n_iter):
        # -------------------------
        # E-step: compute E[phi | r, U, V]
        # -------------------------
        mu = np.sum(U[u_idx] * V[v_idx], axis=1)  # (n_obs,)
        a = r * (mu / np.sqrt(sigma2))            # standardize for N(0,1)

        # lambda = φ(a) / Φ(a), computed stably in log-space
        log_pdf = norm.logpdf(a)
        log_cdf = norm.logcdf(a)  # log Φ(a)
        lam = np.exp(log_pdf - log_cdf)  # (n_obs,)

        # E[phi] for truncated normal (scale back by sqrt(sigma2))
        # mean_phi = mu + r * sqrt(sigma2) * lam
        mean_phi = mu + r * np.sqrt(sigma2) * lam  # (n_obs,)

        # -------------------------
        # M-step: ridge regressions to update U and V
        # -------------------------
        # Update U
        for i in range(n_users):
            ks = obs_by_user[i]
            if ks.size == 0:
                continue
            V_i = V[v_idx[ks]]                    # (#obs_i, d)
            b = (V_i.T @ mean_phi[ks]) * inv_sigma2
            A = (c * I_d) + (V_i.T @ V_i) * inv_sigma2
            U[i] = np.linalg.solve(A, b)

        # Update V
        for j in range(n_movies):
            ks = obs_by_movie[j]
            if ks.size == 0:
                continue
            U_j = U[u_idx[ks]]                    # (#obs_j, d)
            b = (U_j.T @ mean_phi[ks]) * inv_sigma2
            A = (c * I_d) + (U_j.T @ U_j) * inv_sigma2
            V[j] = np.linalg.solve(A, b)

        # -------------------------
        # Log joint objective: log p(R | U,V) + log p(U) + log p(V)
        # where p(r|mu)=Φ(r*mu/sqrt(sigma2)) under probit
        # -------------------------
        mu_new = np.sum(U[u_idx] * V[v_idx], axis=1)
        z = r * (mu_new / np.sqrt(sigma2))
        loglik = np.sum(norm.logcdf(z))

        logprior = -(c / (2.0 * sigma2)) * (np.sum(U * U) + np.sum(V * V))
        ll[t] = loglik + logprior  # (additive constant ignored)

        if (t + 1) % 10 == 0:
            print(f"iter {t+1:3d} | objective (up to const): {ll[t]:.3f}")

    # Part (a) plot: iterations 2 through 100 (i.e., t=1..99)
    iters = np.arange(1, n_iter) + 1  # [2..100] if n_iter=100
    plt.figure()
    plt.plot(iters, ll[1:], linewidth=2)
    plt.xlabel("Iteration")
    plt.ylabel("log p(R, U, V)  (up to additive constant)")
    plt.title("EM objective trace (iterations 2–100)")
    plt.tight_layout()
    plt.show()

    return ll


if __name__ == "__main__":
    # Adjust the path to wherever you put ratings.csv
    ll = em_probit_mf_part_a(
        ratings_csv="ratings.csv",
        d=5,
        c=1.0,
        sigma2=1.0,
        n_iter=100,
        seed=0,
    )