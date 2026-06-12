#! /usr/bin/python3

##--------------------------------------------------------------------\
#   bayesian_optimization_python
#   './bayesian_optimization_python/src/surrogate_models/RBF_network.py'
#   Radial Basis Function network surrogate model for optimization. 
#
#   Note: the underscores before the func names are to differentiate the 
#       layers of optimizer/surrogate modeling
#
#   Author(s): Lauren Linkous 
#   Last update: June 11, 2026
#
#   Interface conventions (shared by all surrogate models):
#       fit(X, Y)
#       predict(X, n_outputs=1) -> (mu, noError)
#       calculate_variance()    -> 1D np.array, len = num points in last predict
#
#   calculate_variance() now returns a distance-inflated estimate instead
#   of zeros. An RBF network with centers at the sample points interpolates
#   the training data (near-zero residuals), so a residual-only estimate
#   would still kill exploration. The estimate grows from the training
#   residual MSE toward the training data variance as the query point
#   moves away from the nearest sample, GP-style.
##--------------------------------------------------------------------\


import numpy as np

class RBFNetwork:
    def __init__(self, kernel='gaussian', epsilon=1.0):
        self.kernel = kernel
        self.epsilon = epsilon
        self.last_predictions  = []
        self.last_X = None
        self.centers = None
        self.weights = None
        self.train_mse = 0.0
        self.y_var = 0.0
        self.is_fitted = False

        
    # configuration check for surrogate models
    # important for AntennCAT surrogate model use. can skip otherwise
    def _check_configuration(self, init_pts, kernel):
        noError, errMsg = self._check_initial_points(init_pts)

        if noError == False: #return first issue
            return noError, errMsg

        noError, errMsg = self._check_kernel(kernel)

        return noError, errMsg
        

    def _check_initial_points(self, init_pts):
        MIN_INIT_POINTS = 1
        errMsg = ""
        noError = True        
        if init_pts < MIN_INIT_POINTS:
            errMsg = "ERROR: minimum required initial points is " + str(MIN_INIT_POINTS)
            noError = False
        return noError, errMsg

    def _check_kernel(self, kernel):
        errMsg = ""
        noError = True        
        if not(kernel == 'gaussian' or kernel == 'multiquadric'):
            errMsg = "WARNING: unrecognized kernel type:" + str(kernel)
            noError = False
        return noError, errMsg
    
    # SM functions
    def _kernel_function(self, x, c):
        if self.kernel == 'gaussian':
            return np.exp(-self.epsilon * np.linalg.norm(x - c) ** 2)
        elif self.kernel == 'multiquadric':
            return np.sqrt(1 + self.epsilon * np.linalg.norm(x - c) ** 2)
        else:
            print("ERROR: Unsupported kernel type in RBFNetwork: " + str(self.kernel))
            

    def _compute_design_matrix(self, X):
        num_samples = X.shape[0]
        num_centers = self.centers.shape[0]
        Phi = np.zeros((num_samples, num_centers))
        for i in range(num_samples):
            for j in range(num_centers):
                Phi[i, j] = self._kernel_function(X[i], self.centers[j])
        return Phi

    def fit(self, X, y):
        if len(X) < 1:
            print("ERROR: at least one initial point needed for this kernel")
            return
        X = np.atleast_2d(X)
        # reshape anchored on the number of SAMPLE rows. the previous
        # y.reshape(y.shape[0], -1) treated each output dimension as a
        # separate row when a single multi-output point was passed in,
        # raising LinAlgError in lstsq (pre-existing bug, hit when
        # num_init_points=1 on a multi-objective function)
        y = np.asarray(y).reshape(X.shape[0], -1)

        self.centers = X
        Phi = self._compute_design_matrix(X)
        self.weights = np.linalg.lstsq(Phi, y, rcond=None)[0]

        # stored for the variance estimate
        residuals = y - np.dot(Phi, self.weights)
        self.train_mse = float(np.mean(residuals ** 2))
        self.y_var = float(np.mean(np.var(y, axis=0)))
        self.is_fitted = True

    def predict(self, X, n_outputs=1):
        noErrors = True
        if not self.is_fitted:
            print("ERROR: RBFNetwork model is not fitted yet")
            noErrors = False
        
        try:
            self.last_X = np.atleast_2d(X)
            Phi = self._compute_design_matrix(self.last_X)
            self.last_predictions = np.dot(Phi, self.weights)
        except Exception as e:
            print("ERROR in RBFNetwork.predict(): " + str(e))
            self.last_predictions = []
            noErrors = False # previously a bare `noErrors` no-op. bug fix.
        return self.last_predictions , noErrors

    def calculate_variance(self):
        #used for calculating expected improvement, but not applying objective func
        # use the last predictions so not calculating everything twice
        if self.last_X is None or len(np.atleast_1d(self.last_predictions)) < 1:
            return np.zeros(0)
        # squared distance to nearest center, per query point
        d2 = np.min(np.sum((self.last_X[:, None, :] - self.centers[None, :, :]) ** 2, axis=2), axis=1)
        # grows from train_mse (at a sample) toward train_mse + y_var (far away)
        growth = 1.0 - np.exp(-self.epsilon * d2)
        return (self.train_mse + self.y_var * growth).ravel()
