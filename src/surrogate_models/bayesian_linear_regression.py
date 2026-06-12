#! /usr/bin/python3

##--------------------------------------------------------------------\
#   bayesian_optimization_python
#   './bayesian_optimization_python/src/surrogate_models/bayesian_linear_regression.py'
#   Bayesian Linear Regression surrogate model for optimization.
#
#   Conjugate Bayesian linear regression (Bayesian ridge) with an
#   optional polynomial feature expansion, implemented with numpy only.
#
#   Unlike the Lagrangian linear/polynomial regression models, this model
#   has a closed-form POSTERIOR over the weights, which yields an analytic,
#   input-dependent predictive variance:
#
#       prior:       w ~ N(0, alpha^-1 * I)
#       likelihood:  y ~ N(Phi(x)^T w, beta^-1)
#       posterior:   w ~ N(m, S),  S = (alpha*I + beta*Phi^T Phi)^-1
#                                  m = beta * S * Phi^T y
#       predictive:  y* ~ N(phi*^T m, 1/beta + phi*^T S phi*)
#
#   The phi*^T S phi* term grows in regions far from (or orthogonal to)
#   the training data, so expected improvement gets a real exploration
#   signal, unlike the zero- or residual-only-variance regression models.
#
#   Author(s): Lauren Linkous, Anthropic Claude
#   Last update: June 11, 2026
#
#   Interface conventions (shared by all surrogate models):
#       fit(X, Y)
#       predict(X, n_outputs=1) -> (mu, noError)
#       calculate_variance()    -> 1D np.array, len = num points in last predict
##--------------------------------------------------------------------\

import numpy as np

class BayesianLinearRegression:
    def __init__(self, degree=1, alpha=1e-2, beta=None):
        # degree: polynomial feature expansion degree. 1 = plain linear.
        #         degree > 1 makes this a Bayesian polynomial regression.
        # alpha:  prior precision on the weights (larger = stronger
        #         shrinkage toward 0, i.e. more regularization).
        # beta:   noise precision (1/observation variance). If None, it is
        #         estimated from the training residuals at fit time.
        self.degree = int(degree)
        self.alpha = float(alpha)
        self.beta_param = beta # None = estimate from residuals
        self.beta = None
        self.mean_weights = None # posterior mean,  (n_features, n_outputs)
        self.S = None            # posterior covariance, (n_features, n_features)
        self.last_X = None
        self.mean = []
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
    def _features(self, X):
        # [1, X, X^2, ..., X^degree] columns. matches the convention used
        # by polynomial_regression.py
        X = np.atleast_2d(X)
        Phi = np.ones((X.shape[0], 1))
        for d in range(1, self.degree + 1):
            Phi = np.hstack((Phi, X ** d))
        return Phi

    def fit(self, X_sample, Y_sample):
        self.X_sample = np.atleast_2d(X_sample)
        self.Y_sample = np.atleast_2d(Y_sample).reshape(self.X_sample.shape[0], -1)

        Phi = self._features(self.X_sample)
        n_features = Phi.shape[1]

        # estimate noise precision from ordinary least squares residuals
        # unless the user fixed it
        if self.beta_param is None:
            w_ols = np.linalg.lstsq(Phi, self.Y_sample, rcond=None)[0]
            residuals = self.Y_sample - Phi.dot(w_ols)
            mse = float(np.mean(residuals ** 2))
            # floor keeps beta finite when the model interpolates exactly
            self.beta = 1.0 / max(mse, 1e-10)
        else:
            self.beta = float(self.beta_param)

        # posterior over the weights (closed form, conjugate prior)
        A = self.alpha * np.eye(n_features) + self.beta * Phi.T.dot(Phi)
        # A is symmetric positive definite by construction (alpha > 0), so
        # the inverse is stable, but fall back to pinv just in case
        try:
            self.S = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            self.S = np.linalg.pinv(A)
        self.mean_weights = self.beta * self.S.dot(Phi.T).dot(self.Y_sample)

        self.is_fitted = True

    def predict(self, X, n_outputs=1):
        noErrors = True
        if not self.is_fitted:
            print("ERROR: BayesianLinearRegression model is not fitted yet")
            noErrors = False

        try:
            self.last_X = np.atleast_2d(X)
            Phi = self._features(self.last_X)
            self.mean = Phi.dot(self.mean_weights) # (n_points, n_outputs)
        except Exception as e:
            print("ERROR in BayesianLinearRegression.predict(): " + str(e))
            self.mean = []
            noErrors = False

        return self.mean, noErrors

    def calculate_variance(self):
        #used for calculating expected improvement, but not applying objective func
        # use the last predictions so not calculating everything twice
        #
        # full predictive variance: observation noise + posterior weight
        # uncertainty projected through the features. one value per point.
        if not self.is_fitted or self.last_X is None:
            return np.zeros(0)
        Phi = self._features(self.last_X)
        # diag(Phi S Phi^T) without forming the full matrix
        model_var = np.sum(Phi.dot(self.S) * Phi, axis=1)
        variance = (1.0 / self.beta) + model_var
        return np.maximum(variance, 0.0).ravel()
