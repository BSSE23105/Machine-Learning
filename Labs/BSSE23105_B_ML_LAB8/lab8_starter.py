import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# ==========================================================
# DATASET (PROVIDED)
# ==========================================================

def load_fraud_data():
    """
    Generates a synthetic dataset for fraud detection.

    The dataset contains two features:
    - Transaction Amount
    - Time Gap Since Last Transaction

    The data consists of:
    - Normally distributed points representing legitimate transactions
    - Uniformly distributed outliers representing fraudulent transactions

    Returns:
    --------
    X : np.ndarray of shape (N, 2)
        Feature matrix

    y : np.ndarray of shape (N,)
        Labels where:
        1  -> Normal transaction
        -1 -> Fraudulent transaction
    """
    np.random.seed(42)

    normally_distributed_transaction_data = np.random.multivariate_normal(mean=[50, 30], cov=[[100, 20],[20, 50]], size=280)

    fraudulent_uniform_transactions_dataset = np.random.uniform(low=[0, 0], high=[120, 100], size=(30,2))

    combined_transactions_matrix_input =  np.vstack([normally_distributed_transaction_data, fraudulent_uniform_transactions_dataset])

    actual_target_labels_for_data  =   np.ones(len(combined_transactions_matrix_input))
    actual_target_labels_for_data[-30:]= -1

    return combined_transactions_matrix_input, actual_target_labels_for_data



# ==========================================================
# ISOLATION FOREST
# ==========================================================




# TASK 1: Implementing a function to train an Isolation Forest model to isolate anomalies by recursively splitting the data.
def fit_isolation_forest(X):
    my_isolation_forest_anomaly_model =  IsolationForest(random_state=42)
    my_isolation_forest_anomaly_model.fit(X)
    return my_isolation_forest_anomaly_model

# TASK 2: Using the trained Isolation Forest model to classify each transaction as normal or anomalous.
def predict_iforest(model, X):
    predicted_labels_from_isolation_forest =   model.predict(X)
    return predicted_labels_from_isolation_forest

# TASK 3: Computing anomaly scores for all data points based on the path length required to isolate them.
def compute_iforest_scores(model, X):
    computed_anomaly_scores_for_each_point =   model.decision_function(X)
    return computed_anomaly_scores_for_each_point




# ==========================================================
# GAUSSIAN (FROM SCRATCH)
# ==========================================================

# TASK 4: Computing the mean and covariance matrix of the dataset to represent its center and spread.
def compute_gaussian_params(X):
    calculated_mean_vector_of_dataset =   np.mean(X, axis=0)
    calculated_covariance_matrix_of_dataset  =  np.cov(X, rowvar=False)
    return calculated_mean_vector_of_dataset, calculated_covariance_matrix_of_dataset

# TASK 5: Computing the probability density of each data point using the multivariate Gaussian distribution.
def gaussian_density(X, mu, sigma):
    number_of_features_dimentions = X.shape[1]
    determinant_of_the_correlation_sigma_matrix =  np.linalg.det(sigma)
    inverse_of_the_correlation_sigma_matrix   =  np.linalg.inv(sigma)
    
    normalization_constant_for_probability  =  1.0 / (((2 * np.pi) ** (number_of_features_dimentions / 2.0)) * (determinant_of_the_correlation_sigma_matrix ** 0.5))
    
    array_of_all_probability_densities  =  np.zeros(X.shape[0])
    for current_row_index_iterator_for_prob in range(X.shape[0]):
        difference_vector_from_mean =   X[current_row_index_iterator_for_prob] - mu
        exponent_calculation_for_current_row =   -0.5 * np.dot(np.dot(difference_vector_from_mean.T, inverse_of_the_correlation_sigma_matrix), difference_vector_from_mean)
        array_of_all_probability_densities[current_row_index_iterator_for_prob] =   normalization_constant_for_probability * np.exp(exponent_calculation_for_current_row)
        
    return array_of_all_probability_densities



# TASK 6: Classifying transactions based on probability values and detecting anomalies using a specified threshold.
def predict_gaussian(X, mu, sigma, threshold):
    probability_densities_from_gaussian  =   gaussian_density(X, mu, sigma)
    final_predicted_class_labels_based_on_prob =  np.ones(X.shape[0])
    final_predicted_class_labels_based_on_prob[probability_densities_from_gaussian < threshold] = -1
    return final_predicted_class_labels_based_on_prob

# ==========================================================
# VISUALIZATION
# ==========================================================

# TASK 7: Plotting the dataset in a two-dimensional space distinguishing normal and anomalous transactions.
def plot_points(X, labels, title):
    color_list = ['blue' if lbl == 1 else 'red' for lbl in labels]
    plt.scatter(X[:,0], X[:,1], c=color_list, edgecolors='white', linewidth=0.8, s=40)
    plt.title(title)

# TASK 8: Visualizing how the methods separate normal data from anomalies in the feature space.
def plot_decision_boundary(model_or_fn, X, method):
    minimum_value_on_x_axis, maximum_value_on_x_axis   =  X[:, 0].min() - 10, X[:, 0].max() + 10
    minimum_value_on_y_axis, maximum_value_on_y_axis  =    X[:, 1].min() - 10, X[:, 1].max() + 10
    coordinate_x_meshgrid_points, coordinate_y_meshgrid_points  =  np.meshgrid(np.linspace(minimum_value_on_x_axis, maximum_value_on_x_axis, 100), np.linspace(minimum_value_on_y_axis, maximum_value_on_y_axis, 100))
    grid_of_all_the_plot_points   =    np.c_[coordinate_x_meshgrid_points.ravel(), coordinate_y_meshgrid_points.ravel()]

    if method == "iforest":
        predicted_z_values_for_contour_plot  =   model_or_fn.predict(grid_of_all_the_plot_points)
    elif method == "gaussian":
        extracted_mu_vector, extracted_sigma_matrix, extracted_density_threshold   =  model_or_fn
        predicted_z_values_for_contour_plot =  predict_gaussian(grid_of_all_the_plot_points, extracted_mu_vector, extracted_sigma_matrix, extracted_density_threshold)
    else:
        return

    predicted_z_values_for_contour_plot =  predicted_z_values_for_contour_plot.reshape(coordinate_x_meshgrid_points.shape)
    plt.contourf(coordinate_x_meshgrid_points, coordinate_y_meshgrid_points, predicted_z_values_for_contour_plot, alpha=0.4, colors=['#ffc2c2', '#ffffff'])





if __name__ == '__main__':
    loaded_features_x_data_from_dataset, loaded_target_y_labels_from_dataset   =   load_fraud_data()

    plt.figure(figsize=(8, 6))
    plot_points(loaded_features_x_data_from_dataset, loaded_target_y_labels_from_dataset, "Original Dataset - True Labels")
    plt.xlabel('Transaction Amount')
    plt.ylabel('Time Gap Since Last Transaction')
    plt.savefig('plot_dataset.png', dpi=150, bbox_inches='tight')
    
    constructed_isolation_forest_model_object   =   fit_isolation_forest(loaded_features_x_data_from_dataset)
    predictions_from_the_isolation_forest_model  =     predict_iforest(constructed_isolation_forest_model_object, loaded_features_x_data_from_dataset)
    computed_scores_from_isolation_forest  =  compute_iforest_scores(constructed_isolation_forest_model_object, loaded_features_x_data_from_dataset)

    estimated_dataset_mu_params_mean_center, estimated_dataset_cov_params_covariance_spread  =   compute_gaussian_params(loaded_features_x_data_from_dataset)
    calculated_probability_densities_array_for_data  =   gaussian_density(loaded_features_x_data_from_dataset, estimated_dataset_mu_params_mean_center, estimated_dataset_cov_params_covariance_spread)
    determined_cut_off_threshold_for_anomalies_5_percent =  np.percentile(calculated_probability_densities_array_for_data, 5)
    predictions_from_our_gaussian_density_model  =  predict_gaussian(loaded_features_x_data_from_dataset, estimated_dataset_mu_params_mean_center, estimated_dataset_cov_params_covariance_spread, determined_cut_off_threshold_for_anomalies_5_percent)

    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plot_decision_boundary(constructed_isolation_forest_model_object, loaded_features_x_data_from_dataset, "iforest")
    plot_points(loaded_features_x_data_from_dataset, predictions_from_the_isolation_forest_model, "Isolation Forest Decisions")
    plt.xlabel('Transaction Amount')
    plt.ylabel('Time Gap Since Last Transaction')
    
    plt.subplot(1, 2, 2)
    plot_decision_boundary((estimated_dataset_mu_params_mean_center, estimated_dataset_cov_params_covariance_spread, determined_cut_off_threshold_for_anomalies_5_percent), loaded_features_x_data_from_dataset, "gaussian")
    plot_points(loaded_features_x_data_from_dataset, predictions_from_our_gaussian_density_model, "Gaussian Density Decisions")
    plt.xlabel('Transaction Amount')
    plt.ylabel('Time Gap Since Last Transaction')
    
    plt.tight_layout()
    plt.savefig('plots.png', dpi=150, bbox_inches='tight')
    plt.show()