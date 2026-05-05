import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
 

def load_ratings(path_csv: str):
    """
    Expects columns: user_id, movie_id, rating   where rating in {+1, -1}.
    Returns:
      u_idx (n_obs,), v_idx (n_obs,), r (n_obs,), n_users, n_movies,
      plus the ID maps (for debugging).
    """
    df = pd.read_csv(path_csv, header=None, names=['user_id', 'movie_id', 'rating'])
    return df


# Define em_algo function
def em_algo(seed, iterations =100):
    
    np.random.seed(seed)
    # Generate each u_i and v_j from Normal (0, 0.1I)
    # Documentation: https://numpy.org/devdocs/reference/random/generated/numpy.random.multivariate_normal.html
    U_em = np.random.multivariate_normal(np.zeros(d), 0.1 * np.eye(d), N)
    V_em = np.random.multivariate_normal(np.zeros(d), 0.1 * np.eye(d), M)

    # Initialize log_likelihood_em
    log_joint_em = []
    for t in range(iterations):
        # E-STEP
        # First, calculate first u_i^T * v_j term using vectorized operation np.einsum()
        lin_predictor = (np.einsum('nd,nd->n', U_em[training_i - 1], V_em[training_j- 1]))
        
        # Create cdf and pdf. Documentation for norm: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html
        pdf_dist = norm.pdf(-1*lin_predictor)
        cdf_dist = norm.cdf(-1*lin_predictor)
        # Use np.where to cover two cases where r_ij = 1 and where r_ij = -1
        posterior_exp = np.where(training_r == 1,
                                 lin_predictor + sigma * (pdf_dist)/(1-cdf_dist),
                                 lin_predictor - sigma * (pdf_dist/cdf_dist))
        
        # M-STEP
        # Update u_i
        u_updated = np.zeros((N,d))
        for i in range(1, N+1):
            # Get rid of all ratings not made by user i themselves through mask
            mask = (training_i == i)
            v_j = V_em[training_j[mask]-1]
            phi_ij = posterior_exp[mask]

            # Split up into terms (using @ for matrix multiplication) and solve with linalg.solve
            # Documentation for linalg.solve: https://numpy.org/devdocs/reference/generated/numpy.linalg.solve.html
            first_u_term = ((1/c) * np.eye(d) + (1/sigma_squared) * (v_j.T @ v_j))
            second_u_term = ((1/sigma_squared) * (v_j.T @ phi_ij))
            u_updated[i-1] = np.linalg.solve(first_u_term, second_u_term)

        # Update v_j
        v_updated = np.zeros((M,d))
        for j in range(1, M + 1):
            ## Get rid of all movies not rate by user through mask
            mask = (training_j == j)
            u_i = U_em[training_i[mask]-1]
            phi_ij = posterior_exp[mask]

            # Solve
            first_v_term = ((1/c) * np.eye(d) + (1/sigma_squared) * (u_i.T @ u_i))
            second_v_term = ((1/sigma_squared) * (u_i.T @ phi_ij))
            v_updated[j-1] = np.linalg.solve(first_v_term, second_v_term)

        U_em = u_updated
        V_em = v_updated

        # Check for convergence with ln p(R, U, V)
        # First, calculate the new inner product of u_i^T and v_j
        updated_lin_predictor = np.einsum('nd, nd -> n', U_em[training_i - 1], V_em[training_j-1])
        log_joint = ((-1/(2*c)) * np.sum(U_em**2)
                     - ((1/(2*c)) * np.sum(V_em**2))
                     + np.sum(norm.logcdf(updated_lin_predictor[training_r == 1]/sigma))
                     + np.sum(np.log(1- norm.cdf(updated_lin_predictor[training_r == -1]/sigma)))
                    )
        log_joint_em.append(log_joint)

    return U_em, V_em, log_joint_em


if __name__ == "__main__":
    path_csv = 'EECS6720_data_hw1/ratings.csv'
    data = load_ratings(path_csv)
    
    N = data['user_id'].max() 
    M = data['movie_id'].max()
    # Set up arrays for vectorized operations, which saves time vs looping over each operatio
    training_i = data['user_id'].values
    training_j = data['movie_id'].values
    training_r = data['rating'].values
    
    # Initialize as instructed
    d = 5
    c = 1
    sigma_squared = 1
    sigma = 1
    
    # Problem 2a) Run algorithm for 100 iterations
    U_em, V_em, log_joint_em = em_algo(seed = 0, iterations = 100)
    # Plot iterations 2 - 100
    plt.figure(figsize = (5,5))
    plt.plot(range(2,101), log_joint_em[1:])
    plt.title('Problem 2(a): EM Algorithm (Iterations 2~100)')
    plt.xlabel('Iteration')
    plt.ylabel('ln p(R, U, V)')
    plt.show()
    
    # Problem 2b) Plot for 100 iteratios using 5 different random starting points
    seeds = [1,2,3,4,5]
    plt.figure(figsize = (5, 5))
    loop_tracker = 1
    for seed in seeds:
        _, _, log_joint_em = em_algo(seed = seed, iterations = 100)
        plt.plot(range(20, 101), log_joint_em[19:], label = f'Starting Run {loop_tracker}')
        loop_tracker +=1

    plt.title('Problem 2(b): EM Algorithm with 5 different starting points (Iterations 20~100)')
    plt.xlabel('Iterations')
    plt.ylabel('ln p (R, U, V)')
    plt.legend()
    plt.show()
    
    # Problem 2c) test data
    test_path_csv = 'EECS6720_data_hw1/ratings_test.csv'
    testdata = load_ratings(test_path_csv)
    testing_i = testdata['user_id'].values
    testing_j = testdata['movie_id'].values
    testing_r = testdata['rating'].values
    
    U_em, V_em, log_joint_em = em_algo(seed = 0, iterations = 100)
    # Use trained parameters U and V to get a better linear predictor
    lin_pred_test = np.einsum('nd,nd->n', U_em[testing_i - 1], V_em[testing_j - 1])
    pred_rating = np.where(lin_pred_test >= 0, 1, -1)
    act_rating = testing_r
    
    # Display confusion matrix
    em_confusion = confusion_matrix(act_rating, pred_rating)
    disp = ConfusionMatrixDisplay(confusion_matrix = em_confusion, display_labels = [-1, 1])
    disp.plot(cmap='Greys', colorbar=False) 
    plt.title('Problem 2(c): Confusion Matrix with Test Data')
    plt.show()
    