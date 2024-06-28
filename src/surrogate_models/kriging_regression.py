#! /usr/bin/python3

##--------------------------------------------------------------------\
#   bayesian_optimization_python
#   './bayesian_optimization_python/src/surrogate_models/kriging_regression.py'
#   Kriging (Gaussian process regression) surrogate model for optimization. 
#
#   Author(s): Lauren Linkous 
#   Last update: June 25, 2024
##--------------------------------------------------------------------\

import numpy as np

class Kriging:
    def __init__(self, length_scale=1.1, noise=1e-10):
        self.length_scale = length_scale
        self.noise = noise
        self.is_fitted = False

    def empirical_variogram(self, X, Y):
        # Calculate empirical variogram from data X and Y
        dists = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))  # Euclidean distance
        variogram = np.abs(Y[:, None, :] - Y[None, :, :]) ** 2 / 2
        return dists, variogram.mean(axis=2)  # Average over the third dimension

    def fit(self, X_sample, Y_sample):
        self.X_sample = np.atleast_2d(X_sample)
        self.Y_sample = np.atleast_2d(Y_sample).reshape(X_sample.shape[0], -1)  # Flatten Y_sample to 2D

        # Calculate empirical variogram
        dists, variogram = self.empirical_variogram(self.X_sample, self.Y_sample)

        # Fit a variogram model to the empirical variogram (simplified example)
        # For example, fitting a linear model to the variogram
        self.variogram_model_params = np.polyfit(dists.flatten(), variogram.flatten(), deg=1)

        # Compute the covariance matrix of the sample points using the variogram model
        self.K = self.variogram_model_params[0] * dists + self.variogram_model_params[1] + self.noise * np.eye(len(X_sample))
        self.K_inv = np.linalg.inv(self.K)

        self.is_fitted = True

    def predict(self, X, out_dims=1):
        if not self.is_fitted:
            print("ERROR: Kriging model is not fitted yet")
        
        X = np.atleast_2d(X)
        
        # Calculate distances between sample points and new points X
        dists_to_sample = np.sqrt(np.sum((self.X_sample[:, None, :] - X[None, :, :]) ** 2, axis=2))
        
        # Use the variogram model to compute covariances
        covariances = self.variogram_model_params[0] * dists_to_sample + self.variogram_model_params[1]
        
        # Compute the weights using the inverse of the covariance matrix
        weights = np.dot(self.K_inv, covariances)
        
        # Compute the predictions
        predictions = np.dot(weights.T, self.Y_sample)
        
        # Estimate the variance of the prediction
        K_ss = np.zeros((len(X), len(X)))  # Placeholder for self-covariance matrix of new points
        for i in range(len(X)):
            for j in range(len(X)):
                K_ss[i, j] = self.variogram_model_params[0] * np.linalg.norm(X[i] - X[j]) + self.variogram_model_params[1]

        cov_prediction = K_ss - np.dot(weights.T, covariances)
        
        return predictions, np.diag(cov_prediction).reshape(-1,1)
