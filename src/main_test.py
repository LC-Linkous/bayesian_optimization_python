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
#import surrogate model kernel
from gaussian_process import GaussianProcess
#import optimizer
from bayesian_optimizer import BayesianOptimization

# import objective function
# single objective
import himmelblau.configs_F as func_configs


class Test():
    def __init__(self):
        LB = [[-5, -5]]           # Lower boundaries
        UB = [[5, 5]]             # Upper boundaries
        OUT_VARS = 1              # Number of output variables (y-values)
        TARGETS = [0]             # Target values for output
        E_TOL = 10 ** -6          # Convergence Tolerance
        MAXIT = 200               # Maximum allowed iterations

        # Objective function dependent variables
        func_F = func_configs.OBJECTIVE_FUNC  # objective function
        constr_F = func_configs.CONSTR_FUNC   # constraint function

        # Bayesian optimizer tuning params
        self.init_num_points = 2 
        xi = 0.01
        n_restarts = 25

        # Gaussian Process vars
        noise = 1e-10
        length_scale = 1.0

        self.best_eval = 9999    # set higher than normal because of the potential for missing the target

        parent = self            # Optional parent class for swarm 
                                            # (Used for passing debug messages or
                                            # other information that will appear 
                                            # in GUI panels)

        self.suppress_output = False    # Suppress the console output of optimizer

        detailedWarnings = False        # Optional boolean for detailed feedback
                                        # (Independent of suppress output. 
                                        #  Includes error messages and warnings)

        self.allow_update = True      # Allow objective call to update state 


        self.gp = GaussianProcess(length_scale=length_scale,noise=noise)  # select the surrogate model
        self.bayesOptimizer = BayesianOptimization(LB, UB, OUT_VARS, TARGETS, E_TOL, MAXIT,
                                                    func_F, constr_F, 
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
         

    def fit_model(self, x, y):
        # call out to parent class to use surrogate model
        self.gp.fit(x,y)
        

    def model_predict(self, x):
        # call out to parent class to use surrogate model
        mu, sigma = self.gp.predict(x)
        return mu, sigma


    def run(self):

        
        # set up the initial sample points  (randomly generated in this example)
        self.bayesOptimizer.initialize_starting_points(self.init_num_points)
        # get the sample points out (to ensure standard formatting)
        x_sample, y_sample = self.bayesOptimizer.get_sample_points()
        # fit GP model.
        self.gp.fit(x_sample, y_sample)


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
