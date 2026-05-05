import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

path = 'EECS6720_data_hw1'

# load data
train_data = pd.read_csv(path + '/ratings.csv', header=None, names=['user', 'movie', 'rating'])
test_data = pd.read_csv(path + '/ratings_test.csv', header=None, names=['user', 'movie', 'rating'])

# convert to 0-indexed
train_data['user'] -= 1
train_data['movie'] -= 1
test_data['user'] -= 1
test_data['movie'] -= 1

num_users = train_data['user'].max() + 1
num_movies = train_data['movie'].max() + 1

# pull out arrays for fast access
user_indices = train_data['user'].values
movie_indices = train_data['movie'].values
train_ratings = train_data['rating'].values.astype(float)



test_user_indices = test_data['user'].values
test_movie_indices = test_data['movie'].values
test_ratings = test_data['rating'].values


# model settings
latent_dim = 5
prior_variance = 1.0
noise_variance = 1.0
noise_std = np.sqrt(noise_variance)

# Problem 3: Gibbs sampler (MCMC)


def log_joint_mcmc(user_vectors, movie_vectors):
    predicted_ratings = np.sum(user_vectors[user_indices] * movie_vectors[movie_indices], axis=1)
    squared_error = np.sum((train_ratings - predicted_ratings) ** 2)
    log_likelihood = -0.5 / noise_variance * squared_error
    log_prior = -0.5 / prior_variance * (np.sum(user_vectors ** 2) + np.sum(movie_vectors ** 2))
    return log_likelihood + log_prior


def gibbs_step(user_vectors, movie_vectors, rng):
    new_user_vectors = np.zeros_like(user_vectors)
    for i in range(num_users):
        user_mask = (user_indices == i)
        if not user_mask.any():
            continue
        movies_rated_by_i = movie_vectors[movie_indices[user_mask]]
        ratings_by_i = train_ratings[user_mask]
        posterior_precision = (1.0 / noise_variance) * (movies_rated_by_i.T @ movies_rated_by_i) + (1.0 / prior_variance) * np.eye(latent_dim)
        posterior_cov = np.linalg.inv(posterior_precision)
        posterior_mean = (1.0 / noise_variance) * posterior_cov @ (movies_rated_by_i.T @ ratings_by_i)
        new_user_vectors[i] = rng.multivariate_normal(posterior_mean, posterior_cov)

    new_movie_vectors = np.zeros_like(movie_vectors)
    for j in range(num_movies):
        movie_mask = (movie_indices == j)
        if not movie_mask.any():
            continue
        users_who_rated_j = new_user_vectors[user_indices[movie_mask]]
        ratings_for_j = train_ratings[movie_mask]
        posterior_precision = (1.0 / noise_variance) * (users_who_rated_j.T @ users_who_rated_j) + (1.0 / prior_variance) * np.eye(latent_dim)
        posterior_cov = np.linalg.inv(posterior_precision)
        posterior_mean = (1.0 / noise_variance) * posterior_cov @ (users_who_rated_j.T @ ratings_for_j)
        new_movie_vectors[j] = rng.multivariate_normal(posterior_mean, posterior_cov)

    return new_user_vectors, new_movie_vectors


print('\nRunning Problem 3 (Gibbs sampler, 1000 iterations) ...')

rng = np.random.default_rng(0)
user_vectors_gibbs = np.zeros((num_users, latent_dim))
movie_vectors_gibbs = np.zeros((num_movies, latent_dim))

log_joint_history_gibbs = []
collected_user_vectors = []
collected_movie_vectors = []

for t in range(1000):
    user_vectors_gibbs, movie_vectors_gibbs = gibbs_step(user_vectors_gibbs, movie_vectors_gibbs, rng)
    log_joint_history_gibbs.append(log_joint_mcmc(user_vectors_gibbs, movie_vectors_gibbs))

    iteration = t + 1
    # collect a sample every 25 iterations after burn-in (starting at iter 100)
    if iteration >= 100 and iteration % 25 == 0:
        collected_user_vectors.append(user_vectors_gibbs.copy())
        collected_movie_vectors.append(movie_vectors_gibbs.copy())

    if iteration % 100 == 0:
        print(f'  iter {iteration}/1000  ln p = {log_joint_history_gibbs[-1]:.2f}')

# Problem 3(b): two plots of the log joint
plt.figure()
plt.plot(range(1, 1001), log_joint_history_gibbs)
plt.xlabel('Iteration')
plt.ylabel('ln p(R, U, V)')
plt.title('Problem 3(b): Log joint (all 1000 iterations)')
plt.tight_layout()
plt.savefig('p3b_all.png', dpi=150)
plt.show()

plt.figure()
plt.plot(range(100, 1001), log_joint_history_gibbs[99:])
plt.xlabel('Iteration')
plt.ylabel('ln p(R, U, V)')
plt.title('Problem 3(b): Log joint (iterations 100-1000)')
plt.tight_layout()
plt.savefig('p3b_100to1000.png', dpi=150)
plt.show()

# Problem 3(c): Monte Carlo prediction on test set
# Average u_i^T v_j over collected samples to estimate E[r_ij | R]
all_user_samples = np.array(collected_user_vectors)   # shape (num_samples, num_users, latent_dim)
all_movie_samples = np.array(collected_movie_vectors)  # shape (num_samples, num_movies, latent_dim)

print(f'\nCollected {len(all_user_samples)} samples for MC estimate')

# dot product for each sample and each test pair
dot_products_per_sample = np.sum(
    all_user_samples[:, test_user_indices, :] * all_movie_samples[:, test_movie_indices, :], axis=2
)

# average over samples to get the MC estimate of E[r_ij | R]
mc_expected_rating = dot_products_per_sample.mean(axis=0)
mc_predictions = np.where(mc_expected_rating > 0, 1, -1)

true_neg_3 = np.sum((test_ratings == -1) & (mc_predictions == -1))
false_pos_3 = np.sum((test_ratings == -1) & (mc_predictions == 1))
false_neg_3 = np.sum((test_ratings == 1) & (mc_predictions == -1))
true_pos_3 = np.sum((test_ratings == 1) & (mc_predictions == 1))

print('\nProblem 3(c) Confusion Matrix:')
print(f'             Pred -1   Pred +1')
print(f'True -1 :    {true_neg_3:5d}     {false_pos_3:5d}')
print(f'True +1 :    {false_neg_3:5d}     {true_pos_3:5d}')
print(f'Accuracy: {(true_pos_3 + true_neg_3) / len(test_ratings):.3f}')

fig, ax = plt.subplots(figsize=(4, 3))
ax.axis('off')
confusion_table_3 = ax.table(cellText=[['', 'Pred -1', 'Pred +1'],
                                       ['True -1', true_neg_3, false_pos_3],
                                       ['True +1', false_neg_3, true_pos_3]],
                             loc='center', cellLoc='center')
confusion_table_3.scale(1.4, 2.0)
ax.set_title('Problem 3(c): Confusion Matrix')
plt.tight_layout()
plt.savefig('p3c.png', dpi=150)
plt.show()
