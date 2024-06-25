#! /usr/bin/python3

##--------------------------------------------------------------------\
#   bayesian_optimization_python
#   './bayesian_optimization_python/src/main_test_graph.py'
#   Guassian Process surrogate model for optimization. 
#
#   Author(s): Lauren Linkous 
#   Last update: June 24, 2024
##--------------------------------------------------------------------\

import numpy as np

class GaussianProcess:
    def __init__(self, length_scale=1.1, noise=1e-10):
        self.length_scale = length_scale
        self.noise = noise
        self.is_fitted = False

    def rbf_kernel(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        dists = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * dists / self.length_scale**2)

    def fit(self, X_sample, Y_sample):
        self.X_sample = np.atleast_2d(X_sample)
        self.Y_sample = np.atleast_2d(Y_sample)
        self.K = self.rbf_kernel(self.X_sample, self.X_sample) + self.noise * np.eye(len(self.X_sample))
        self.K_inv = np.linalg.inv(self.K)
        self.is_fitted = True

    def predict(self, X, out_dims=2):
        if not self.is_fitted:
            raise ValueError("GaussianProcess model is not fitted yet")
        X = np.atleast_2d(X)
        K_s = self.rbf_kernel(self.X_sample, X)
        K_ss = self.rbf_kernel(X, X) + self.noise * np.eye(len(X))

        ysample = self.Y_sample.reshape(-1, out_dims)
        mu_s = K_s.T.dot(self.K_inv).dot(ysample)
        # mu_s = K_s.T.dot(self.K_inv).dot(self.Y_sample)
        cov_s = K_ss - K_s.T.dot(self.K_inv).dot(K_s)
        return mu_s.ravel(), np.diag(cov_s)

