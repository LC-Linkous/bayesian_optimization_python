#! /usr/bin/python3

##--------------------------------------------------------------------\
#   bayesian_optimization_python
#   './bayesian_optimization_python/src/surrogate_models/decision_tree_regression.py'
#   Decision Tree Regression surrogate model for optimization. 
#
#   Author(s): Lauren Linkous 
#   Last update: June 11, 2026
#
#   Interface conventions (shared by all surrogate models):
#       fit(X, Y)
#       predict(X, n_outputs=1) -> (mu, noError)
#       calculate_variance()    -> 1D np.array, len = num points in last predict
#
#   Leaves now store the variance of their training Y values in addition
#   to the mean, so calculate_variance() returns real per-leaf uncertainty
#   instead of zeros (zeros collapse expected improvement to greedy
#   exploitation with no exploration).
##--------------------------------------------------------------------\


import numpy as np

class DecisionTreeRegression:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None
        self.last_predictions = []
        self.last_variances = []
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
        X = np.atleast_2d(X)
        Y = np.asarray(Y)
        self.tree = self._build_tree(X, Y, depth=0)
        self.is_fitted = True

    def _make_leaf(self, Y):
        # Leaf node: mean prediction plus variance of the training Y
        # values that landed in this leaf (averaged over output dims).
        return {'leaf': True,
                'value': np.mean(Y, axis=0),
                'variance': float(np.mean(np.var(Y, axis=0)))}

    def _build_tree(self, X, Y, depth):
        if depth == self.max_depth or self._check_uniform(Y):
            return self._make_leaf(Y)

        num_features = X.shape[1]
        best_feature, best_split_value = None, None
        best_mse = np.inf

        for feature in range(num_features):
            unique_values = np.unique(X[:, feature])
            for value in unique_values:
                left_mask = X[:, feature] <= value
                right_mask = X[:, feature] > value
                if np.sum(left_mask) > 0 and np.sum(right_mask) > 0:
                    left_y = Y[left_mask]
                    right_y = Y[right_mask]
                    mse = self._calculate_mse(left_y, right_y)
                    if mse < best_mse:
                        best_mse = mse
                        best_feature = feature
                        best_split_value = value

        if best_feature is None:
            return self._make_leaf(Y)

        left_mask = X[:, best_feature] <= best_split_value
        right_mask = X[:, best_feature] > best_split_value
        left_subtree = self._build_tree(X[left_mask], Y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], Y[right_mask], depth + 1)

        return {'leaf': False,
                'feature': best_feature,
                'split_value': best_split_value,
                'left': left_subtree,
                'right': right_subtree}

    def _check_uniform(self, Y):
        return np.all(Y == Y[0])

    def _calculate_mse(self, left_y, right_y):
        mse_left = np.var(left_y) * len(left_y)
        mse_right = np.var(right_y) * len(right_y)
        return mse_left + mse_right

    def predict(self, X, n_outputs=1):
        noErrors = True
        # this is applying the objective function for the surrogate model
        if not self.is_fitted:
            print("ERROR: DecisionTreeRegression model is not fitted yet")
            noErrors = False
        X = np.atleast_2d(X)  # Ensure X is 2D
        try:        
            leaves = [self._traverse_tree(x, self.tree) for x in X]
            self.last_predictions = np.array([leaf['value'] for leaf in leaves])
            self.last_variances = np.array([leaf['variance'] for leaf in leaves])
        except Exception as e:
            print("ERROR in DecisionTreeRegression.predict(): " + str(e))
            self.last_predictions = []
            self.last_variances = []
            noErrors = False
        return self.last_predictions, noErrors
    
    def calculate_variance(self):
        #used for calculating expected improvement, but not applying objective func
        # use the last predictions so not calculating everything twice
        return np.asarray(self.last_variances, dtype=float).ravel()

    def _traverse_tree(self, x, node):
        if node['leaf']:
            return node
        if x[node['feature']] <= node['split_value']:
            return self._traverse_tree(x, node['left'])
        else:
            return self._traverse_tree(x, node['right'])
