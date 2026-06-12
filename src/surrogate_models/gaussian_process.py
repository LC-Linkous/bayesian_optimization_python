#! /usr/bin/python3

##--------------------------------------------------------------------\
#   bayesian_optimization_python
#   './bayesian_optimization_python/src/surrogate_models/gaussian_process.py'
#   Guassian Process surrogate model for optimization. 
#
#   Author(s): Lauren Linkous 
#   Last update: June 11, 2026
#
#   Interface conventions (shared by all surrogate models):
#       fit(X, Y)
#       predict(X, n_outputs=1) -> (mu, noError)
#       calculate_variance()    -> 1D np.array, len = num points in last predict
#   No exceptions escape predict(); errors are reported via the
#   (value, noError) convention so the optimizer state machine is never
#   interrupted mid-run.
##--------------------------------------------------------------------\

import numpy as np

class GaussianProcess:
    def __init__(self, length_scale=1.1, noise=1e-10):
        self.length_scale = length_scale
        self.K_s = None
        self.K_ss = None
        self.noise = noise
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
    def rbf_kernel(self, X1, X2):
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        dists = np.sum((X1[:, None, :] - X2[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * dists / self.length_scale**2)

    def _stable_inverse(self, K):
        # Cholesky-based inverse with escalating jitter.
        # As Bayesian optimization clusters samples, K goes near-singular
        # and np.linalg.inv() becomes unstable or raises. Inverting the
        # triangular Cholesky factor is better conditioned, and the jitter
        # retry recovers from numerically non-PD matrices instead of
        # failing late in a long (e.g. simulation-driven) run.
        jitter = 0.0
        for _ in range(6):
            try:
                L = np.linalg.cholesky(K + jitter * np.eye(len(K)))
                L_inv = np.linalg.inv(L) # triangular, well conditioned
                return L_inv.T.dot(L_inv)
            except np.linalg.LinAlgError:
                jitter = max(jitter * 10, 1e-10)
        # last resort: pseudo-inverse never raises
        return np.linalg.pinv(K)

    def fit(self, X_sample, Y_sample):
        self.X_sample = np.atleast_2d(X_sample)
        self.Y_sample = np.atleast_2d(Y_sample)
        self.K = self.rbf_kernel(self.X_sample, self.X_sample) + self.noise * np.eye(len(self.X_sample))
        self.K_inv = self._stable_inverse(self.K)
        self.is_fitted = True

    def predict(self, X, n_outputs=1):
        noErrors = True
        if not self.is_fitted:
            print("ERROR: GaussianProcess model is not fitted yet")
            noErrors = False
        X = np.atleast_2d(X)
        try:
            self.K_s = self.rbf_kernel(self.X_sample, X)
            self.K_ss = self.rbf_kernel(X, X) + self.noise * np.eye(len(X))

            ysample = self.Y_sample.reshape(-1, n_outputs)
            mu_s = self.K_s.T.dot(self.K_inv).dot(ysample)
            mu_s = mu_s.ravel()
        except Exception as e:
            print("ERROR in GaussianProcess.predict(): " + str(e))
            mu_s = []
            noErrors = False
        return mu_s, noErrors
    

    def calculate_variance(self):
        #used for calculating expected improvement, but not applying objective func
        # use the last predictions so not calculating everything twice
        cov_s = self.K_ss - self.K_s.T.dot(self.K_inv).dot(self.K_s) 
        # clip at 0. floating point error can produce small negatives
        return np.maximum(np.diag(cov_s), 0.0).ravel()
