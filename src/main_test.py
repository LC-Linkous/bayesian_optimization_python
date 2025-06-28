#! /usr/bin/python3

##--------------------------------------------------------------------\
#   bayesian_optimization_python
#   './bayesian_optimization_python/src/main_test.py'
#   Test function/example for using Bayesian Optimizer class in 
#       'bayesian_optimizer.py', and the Gaussian Process surrogate
#       model in 'gaussian_process.py'.
#   Format updates are for integration in the AntennaCAT GUI.
#
#   Author(s): Lauren Linkous
#   Last update: June 28, 2025
##--------------------------------------------------------------------\


import pandas as pd
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
from surrogate_models.matern_process import MaternProcess
from surrogate_models.lagrangian_linear_regression import LagrangianLinearRegression
from surrogate_models.lagrangian_polynomial_regression import LagrangianPolynomialRegression

# OBJECTIVE FUNCTION
#import one_dim_x_test.configs_F as func_configs     # single objective, 1D input
#import himmelblau.configs_F as func_configs         # single objective, 2D input
import lundquist_3_var.configs_F as func_configs    # multi objective function

class Test():
    def __init__(self):
        # OBJECTIVE FUNCTION CONFIGURATIONS FROM FILE

        LB = func_configs.LB                    # Lower boundaries
        UB = func_configs.UB                    # Upper boundaries
        IN_VARS = func_configs.IN_VARS          # Number of input variables (x-values)
        OUT_VARS = func_configs.OUT_VARS        # Number of output variables (y-values)
        GLOBAL_MIN = func_configs.GLOBAL_MIN    # Global minima, if they exist
        TARGETS = func_configs.TARGETS          # Target values for output

        # target format. TARGETS = [0, ...] 

        # threshold is same dims as TARGETS
        # 0 = use target value as actual target. value should EQUAL target
        # 1 = use as threshold. value should be LESS THAN OR EQUAL to target
        # 2 = use as threshold. value should be GREATER THAN OR EQUAL to target
        #DEFAULT THRESHOLD
        THRESHOLD = np.zeros_like(TARGETS) 
        #THRESHOLD = np.ones_like(TARGETS)
        #THRESHOLD = [0, 1, 0]


        # Objective function dependent variables
        self.func_F = func_configs.OBJECTIVE_FUNC  # objective function
        self.constr_F = func_configs.CONSTR_FUNC   # constraint function

        # OPTIMIZER VARIABLES        
        TOL = 10 ** -6                  # Convergence Tolerance. For Sweep, this should be a larger value
        MAXIT = 1000                      # Maximum allowed iterations

        # handling multiple types of graphs
        self.in_vars = IN_VARS
        self.out_vars = OUT_VARS


        
        #default params
        # Bayesian optimizer tuning params
        xi = 0.01
        n_restarts = 25

        # using a variable for options for better debug messages
        SM_OPTION = 2           # 0 = RBF, 1 = Gaussian Process,  2 = Kriging,
                                # 3 = Polynomial Regression, 4 = Polynomial Chaos Expansion, 
                                # 5 = KNN regression, 6 = Decision Tree Regression
                                # 7 = Matern, 8 = Lagrangian Linear Regression
                                # 9 = Lagrangian Polynomial Regression



        # SURROGATE MODEL VARS
        if SM_OPTION == 0:
            # RBF Network vars
            RBF_kernel  = 'gaussian' #options: 'gaussian', 'multiquadric'
            RBF_epsilon = 1.0
            num_init_points = 1
            sm_bayes = RBFNetwork(kernel=RBF_kernel, epsilon=RBF_epsilon)  
            noError, errMsg = sm_bayes._check_configuration(num_init_points, RBF_kernel)

        elif SM_OPTION == 1:
            # Gaussian Process vars
            GP_noise = 1e-10
            GP_length_scale = 1.0
            num_init_points = 1
            sm_bayes = GaussianProcess(length_scale=GP_length_scale,noise=GP_noise) 
            noError, errMsg = sm_bayes._check_configuration(num_init_points)

        elif SM_OPTION == 2:
            # Kriging vars
            K_noise = 1e-10
            K_length_scale = 1.0   
            num_init_points = 2 
            sm_bayes = Kriging(length_scale=K_length_scale, noise=K_noise)
            noError, errMsg = sm_bayes._check_configuration(num_init_points)

        elif SM_OPTION == 3:
            # Polynomial Regression vars
            PR_degree = 5
            num_init_points = 1
            sm_bayes = PolynomialRegression(degree=PR_degree)
            noError, errMsg = sm_bayes._check_configuration(num_init_points)

        elif SM_OPTION == 4:
            # Polynomial Chaos Expansion vars
            PC_degree = 5 
            num_init_points = 1
            sm_bayes = PolynomialChaosExpansion(degree=PC_degree)
            noError, errMsg = sm_bayes._check_configuration(num_init_points)

        elif SM_OPTION == 5:
            # KNN regression vars
            KNN_n_neighbors=3
            KNN_weights='uniform'  #options: 'uniform', 'distance'
            num_init_points = 1
            sm_bayes = KNNRegression(n_neighbors=KNN_n_neighbors, weights=KNN_weights)
            noError, errMsg = sm_bayes._check_configuration(num_init_points)

        elif SM_OPTION == 6:
            # Decision Tree Regression vars
            DTR_max_depth = 5  # options: ints
            num_init_points = 1
            sm_bayes = DecisionTreeRegression(max_depth=DTR_max_depth)
            noError, errMsg = sm_bayes._check_configuration(num_init_points)

        elif SM_OPTION == 7:
            # Matern Process vars
            DTR_max_depth = 1  # options: ints
            num_init_points = 1
            MP_length_scale = 1.1
            MP_noise = 1e-10
            MP_nu = 3/2
            sm_bayes = MaternProcess(length_scale=MP_length_scale, noise=MP_noise, nu=MP_nu)
            noError, errMsg = sm_bayes._check_configuration(num_init_points)

        elif SM_OPTION == 8:
            # Lagrangian penalty linear regression vars
            num_init_points = 2
            LLReg_noise = 1e-10
            LLReg_constraint_degree=1
            sm_bayes = LagrangianLinearRegression(noise=LLReg_noise, constraint_degree=LLReg_constraint_degree)
            noError, errMsg = sm_bayes._check_configuration(num_init_points)

        elif SM_OPTION == 9:
            # Lagrangian penalty polynomial regression vars
            num_init_points = 2
            LPReg_degree = 5
            LPReg_noise = 1e-10
            LPReg_constraint_degree = 3
            sm_bayes = LagrangianPolynomialRegression(degree=LPReg_degree, noise=LPReg_noise, constraint_degree=LPReg_constraint_degree)
            noError, errMsg = sm_bayes._check_configuration(num_init_points)



    
        if noError == False:
            print("ERROR in main_test.py. Incorrect surrogate model configuration")
            print(errMsg)
            return

        self.best_eval = 3       # set higher than normal because of the potential for missing the target
        parent = self            # Optional parent class for swarm 
                                            # (Used for passing debug messages or
                                            # other information that will appear 
                                            # in GUI panels)
        self.suppress_output = False    # Suppress the console output of optimizer
        self.allow_update = True        # Allow objective call to update state 
        evaluate_threshold = True       # use target or threshold. True = THRESHOLD, False = EXACT TARGET

        # Constant variables
        opt_params = {'XI': [xi],                   # exploration float
                    'NUM_RESTARTS': [n_restarts],   # number of predition restarts
                    'INIT_PTS': [num_init_points],  # initial number of samples
                    'SM_MODEL': [sm_bayes]}         # the surrogate model class object


        opt_df = pd.DataFrame(opt_params)
        self.bayesOptimizer = BayesianOptimization(LB, UB, TARGETS, TOL, MAXIT,
                                self.func_F, self.constr_F, 
                                opt_df,
                                parent=parent, 
                                evaluate_threshold=evaluate_threshold, obj_threshold=THRESHOLD)  


    def debug_message_printout(self, txt):
        if txt is None:
            return
        # sets the string as it gets it
        curTime = time.strftime("%H:%M:%S", time.localtime())
        msg = "[" + str(curTime) +"] " + str(txt)
        print(msg)

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
