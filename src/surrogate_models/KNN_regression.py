#! /usr/bin/python3

##--------------------------------------------------------------------\
#   bayesian_optimization_python
#   './bayesian_optimization_python/src/surrogate_models/KNN_regression.py'
#   K-Nearest Neighbors surrogate model for optimization. 
#
#   Author(s): Lauren Linkous 
#   Last update: June 11, 2026
#
#   Interface conventions (shared by all surrogate models):
#       fit(X, Y)
#       predict(X, n_outputs=1) -> (mu, noError)
#       calculate_variance()    -> 1D np.array, len = num points in last predict
#
#   calculate_variance() now returns the (weighted) variance of the k
#   neighbors' Y values rather than zeros. Zero variance collapsed the
#   expected improvement acquisition to pure greedy exploitation of the
#   surrogate mean (no exploration), which is the main reason this model
#   underperformed on the Himmelblau example.
##--------------------------------------------------------------------\

import numpy as np

class KNNRegression:
    def __init__(self, n_neighbors=5, weights='uniform'):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.X_sample = None
        self.Y_sample = None
        self.last_predictions = None
        self.last_neighbor_y = None
        self.last_weights = None
        self.is_fitted = False

    # configuration check for surrogate models
    # important for AntennCAT surrogate model use. can skip otherwise
    def _check_configuration(self, init_pts):
        noError, errMsg = self._check_initial_points(init_pts)
        return noError, errMsg
        
    def _check_initial_points(self, init_pts):
        MIN_INIT_POINTS = 1
        errMsg = ""
        noError = True        
        if init_pts < MIN_INIT_POINTS:
            errMsg = "ERROR: minimum required initial points is " + str(MIN_INIT_POINTS)
            noError = False
        return noError, errMsg

    
    # SM functions
    def fit(self, X, Y):
        self.X_sample = np.atleast_2d(X)
        self.Y_sample = np.asarray(Y).reshape(self.X_sample.shape[0], -1)  # Ensure Y_sample is 2D, rows = samples
        self.is_fitted = True

    def predict(self, X, n_outputs=1):
        noErrors = True
        if not self.is_fitted:
            print("ERROR: KNNRegression model is not fitted yet")
            noErrors = False
            
        X = np.atleast_2d(X)
        self.X_sample = np.atleast_2d(self.X_sample)

        try:
            if np.any(np.isnan(X)) or np.any(np.isnan(self.X_sample)):
                print("WARNING: Input data contains NaN values")

            # Compute distances
            distances = np.sqrt(np.sum((X[:, np.newaxis, :] - self.X_sample[np.newaxis, :, :]) ** 2, axis=-1))
            distances += 1e-45  # Add a small epsilon to avoid division by zero

            # Find indices of nearest neighbors
            # n_neighbors may exceed available samples early in a run
            k = min(self.n_neighbors, self.X_sample.shape[0])
            nearest_indices = np.argsort(distances, axis=1)[:, :k]

            # Calculate weights
            if self.weights == 'uniform':
                weights = np.ones_like(nearest_indices, dtype=float)
            elif self.weights == 'distance':
                weights = 1.0 / distances[np.arange(distances.shape[0])[:, None], nearest_indices]
            else:
                print("ERROR: Unsupported weight type in KNNRegression")
                weights = np.ones_like(nearest_indices, dtype=float)

            # Normalize weights
            weights_sum = np.sum(weights, axis=1, keepdims=True)
            weights_normalized = weights / weights_sum

            # Predict using weighted average of nearest neighbors
            nearest_y = self.Y_sample[nearest_indices]

            # Reshape weights_normalized for broadcasting
            weights_normalized = weights_normalized[:, :, np.newaxis]

            # Compute predictions
            self.last_predictions = np.sum(weights_normalized * nearest_y, axis=1)

            # stored for calculate_variance()
            self.last_neighbor_y = nearest_y
            self.last_weights = weights_normalized
        except Exception as e:
            print("ERROR in KNNRegression.predict(): " + str(e))
            self.last_predictions = []
            noErrors = False
            
        return self.last_predictions, noErrors

    def calculate_variance(self):
        #used for calculating expected improvement, but not applying objective func
        # use the last predictions so not calculating everything twice
        #
        # weighted variance of the k neighbors' Y values around the
        # weighted mean: a real, nearly-free local disagreement estimate.
        # averaged across output dimensions to return one value per point.
        if self.last_neighbor_y is None or len(np.atleast_1d(self.last_predictions)) < 1:
            return np.zeros(0)
        mean = np.atleast_2d(self.last_predictions)[:, np.newaxis, :] # (n, 1, out)
        sq_dev = (self.last_neighbor_y - mean) ** 2                   # (n, k, out)
        var_per_out = np.sum(self.last_weights * sq_dev, axis=1)      # (n, out)
        return np.mean(var_per_out, axis=1).ravel()                   # (n,)
