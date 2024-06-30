#! /usr/bin/python3

##--------------------------------------------------------------------\
#   bayesian_optimization_python
#   './bayesian_optimization_python/src/main_test.py'
#   Test function/example for using Bayesian Optimizer class in 
#       'bayesian_optimizer.py', and the Gaussian Process surrogate
#       model in 'gaussian_process.py'.
#   Format updates are for integration in the AntennaCAT GUI.
#
#   Author(s): Lauren Linkous, Jonathan Lundquist
#   Last update: June 24, 2024
##--------------------------------------------------------------------\


import numpy as np
import time

# OPTIMIZER
from bayesian_optimizer import BayesianOptimization

# SURROGATE MODEL
from surrogate_models.RBF_network import RBFNetwork
from surrogate_models.gaussian_process import GaussianProcess
from surrogate_models.kriging_regression import Kriging
from surrogate_models.polynomial_regression import PolynomialRegression
from surrogate_models.polynomial_chaos_expansion import PolynomialChaosExpansion
from surrogate_models.KNN_regression import KNNRegression
from surrogate_models.decision_tree_regression import DecisionTreeRegression

# OBJECTIVE FUNCTION
#import one_dim_x_test.configs_F as func_configs     # single objective, 1D input
import himmelblau.configs_F as func_configs         # single objective, 2D input
#import lundquist_3_var.configs_F as func_configs    # multi objective function

class Test():
    def __init__(self):
        # OBJECTIVE FUNCTION CONFIGURATIONS FROM FILE

        LB = func_configs.LB                    # Lower boundaries
        UB = func_configs.UB                    # Upper boundaries
        IN_VARS = func_configs.IN_VARS          # Number of input variables (x-values)
        OUT_VARS = func_configs.OUT_VARS        # Number of output variables (y-values)
        TARGETS = func_configs.TARGETS          # Target values for output
        GLOBAL_MIN = func_configs.GLOBAL_MIN    # Global minima, if they exist

        # Objective function dependent variables
        self.func_F = func_configs.OBJECTIVE_FUNC  # objective function
        self.constr_F = func_configs.CONSTR_FUNC   # constraint function

        # OPTIMIZER VARIABLES        
        E_TOL = 10 ** -6                  # Convergence Tolerance. For Sweep, this should be a larger value
        MAXIT = 1000                      # Maximum allowed iterations

        # handling multiple types of graphs
        self.in_vars = IN_VARS
        self.out_vars = OUT_VARS

        # Bayesian optimizer tuning params
        init_num_points = 2 
        xi = 0.01
        n_restarts = 25

        # SURROGATE MODEL VARS
        # RBF Network vars
        RBF_kernel  = 'gaussian' #options: 'gaussian', 'multiquadric'
        RBF_epsilon = 1.0
        # Gaussian Process vars
        GP_noise = 1e-10
        GP_length_scale = 1.0
        # Kriging vars
        K_noise = 1e-10
        K_length_scale = 1.0        
        # Polynomial Regression vars
        PR_degree = 5
        # Polynomial Chaos Expansion vars
        PC_degree = 5 
        # KNN regression vars
        KNN_n_neighbors=3
        KNN_weights='uniform'  #options: 'uniform', 'distance'
        # Decision Tree Regression vars
        DTR_max_depth = 5  # options: ints

        self.best_eval = 9999    # set higher than normal because of the potential for missing the target

        parent = self            # Optional parent class for swarm 
                                            # (Used for passing debug messages or
                                            # other information that will appear 
                                            # in GUI panels)

        self.suppress_output = False    # Suppress the console output of optimizer

        detailedWarnings = False        # Optional boolean for detailed feedback
                                        # (Independent of suppress output. 
                                        #  Includes error messages and warnings)

        self.allow_update = True        # Allow objective call to update state 

        #self.sm = RBFNetwork(kernel=RBF_kernel, epsilon=RBF_epsilon)       
        #self.sm = Kriging(length_scale=K_length_scale, noise=K_noise)
        self.sm = GaussianProcess(length_scale=GP_length_scale,noise=GP_noise)  # select the surrogate model
        #self.sm = PolynomialRegression(degree=PR_degree)
        #self.sm = PolynomialChaosExpansion(degree=PC_degree)
        #self.sm = KNNRegression(n_neighbors=KNN_n_neighbors, weights=KNN_weights)
        #self.sm = DecisionTreeRegression(max_depth=DTR_max_depth)

        self.bayesOptimizer = BayesianOptimization(LB, UB, OUT_VARS, TARGETS, E_TOL, MAXIT,
                                                    self.func_F, self.constr_F, 
                                                    init_points=init_num_points, 
                                                    xi = xi, n_restarts=n_restarts, 
                                                    parent=parent, detailedWarnings=detailedWarnings)


    def debug_message_printout(self, txt):
        if txt is None:
            return
        # sets the string as it gets it
        curTime = time.strftime("%H:%M:%S", time.localtime())
        msg = "[" + str(curTime) +"] " + str(txt)
        print(msg)


    def record_params(self):
        # this function is called from particle_swarm.py to trigger a write to a log file
        # running in the AntennaCAT GUI to record the parameter iteration that caused an error
        pass
         

    # SURROGATE MODEL FUNCS
    def fit_model(self, x, y):
        # call out to parent class to use surrogate model
        self.sm.fit(x,y)
        

    def model_predict(self, x):
        # call out to parent class to use surrogate model
        mu, sigma = self.sm.predict(x, self.out_vars)
        return mu, sigma


    def run(self):        

        # instantiation of particle swarm optimizer 
        while not self.bayesOptimizer.complete():
            # current step() proposes new location, and calculates global best
            self.bayesOptimizer.step()

            # # call the next point sample when it is allowed to update
            # # return control to the optimizer
            self.bayesOptimizer.call_objective(self.allow_update)

            iter, eval = self.bayesOptimizer.get_convergence_data()
            if (eval < self.best_eval) and (eval != 0):
                self.best_eval = eval
            if self.suppress_output:
                if iter%10 ==0: #print out every 10th iteration update
                    print("Iteration")
                    print(iter)
                    print("Best Eval")
                    print(self.best_eval)

        print("Optimized Solution")
        print(self.bayesOptimizer.get_optimized_soln())
        print("Optimized Outputs")
        print(self.bayesOptimizer.get_optimized_outs())


if __name__ == "__main__":
    tg = Test()
    tg.run()
