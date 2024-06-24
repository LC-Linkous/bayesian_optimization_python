#! /usr/bin/python3

##--------------------------------------------------------------------\
#   bayesian_optimization_python
#   './bayesian_optimization_python/src/main_test_graph.py'
#   Test function/example for using Bayesian Optimizer class in 
#       'bayesian_optimizer.py', and the Gaussian Process surrogate
#       model in 'gaussian_process.py'.
#   Format updates are for integration in the AntennaCAT GUI.
#   This version builds from 'main_test.py' to include a 
#       matplotlib plot of objective function and surrogate model
#
#   Author(s): Lauren Linkous, Jonathan Lundquist
#   Last update: June 19, 2024
##--------------------------------------------------------------------\


import numpy as np
import time
import matplotlib.pyplot as plt
import numpy as np
import time
#import surrogate model kernel
from gaussian_process import GaussianProcess
#import optimizer
from bayesian_optimizer import BayesianOptimization

# import objective function
# single objective
import himmelblau.configs_F as func_configs


class TestGraph():
    def __init__(self):


        LB = func_configs.LB                    # Lower boundaries
        UB = func_configs.UB                    # Upper boundaries
        IN_VARS = func_configs.IN_VARS          # Number of input variables (x-values)
        OUT_VARS = func_configs.OUT_VARS        # Number of output variables (y-values)
        TARGETS = func_configs.TARGETS          # Target values for output
        GLOBAL_MIN = func_configs.GLOBAL_MIN    # Global minima, if they exist

        E_TOL = 10 ** -6                 # Convergence Tolerance. For Sweep, this should be a larger value
        MAXIT = 200                      # Maximum allowed iterations

        # Objective function dependent variables
        self.func_F = func_configs.OBJECTIVE_FUNC  # objective function
        self.constr_F = func_configs.CONSTR_FUNC   # constraint function


        # Bayesian optimizer tuning params
        self.init_num_points = 15 
        xi = 0.01
        n_restarts = 25

        # Gaussian Process vars
        noise = 1e-10
        length_scale = 1.0

        #plotting vars - make sure plots and samples match
        self.mesh_sample_dim = 10
        self.lbound = LB
        self.ubound = UB
        # Swarm vars
        self.best_eval = 9999           # set higher than normal because of the potential for missing the target

        parent = self                   # Optional parent class for swarm 
                                        # (Used for passing debug messages or
                                        # other information that will appear 
                                        # in GUI panels)

        self.suppress_output = False    # Suppress the console output of particle swarm

        detailedWarnings = False        # Optional boolean for detailed feedback
                                        # (Independent of suppress output. 
                                        #  Includes error messages and warnings)

        self.allow_update = True        # Allow objective call to update state 


        self.gp = GaussianProcess(length_scale=length_scale,noise=noise)  # select the surrogate model
        self.bayesOptimizer = BayesianOptimization(LB, UB, OUT_VARS, TARGETS, E_TOL, MAXIT,
                                                    self.func_F, self.constr_F, 
                                                    xi = xi, n_restarts=n_restarts,
                                                    parent=parent, detailedWarnings=detailedWarnings)  

        # Matplotlib setup
        # Initialize plot
        self.fig = plt.figure(figsize=(10, 5))
        # objective function and sample points
        self.ax1 = self.fig.add_subplot(131, projection='3d')
        # aquisition function
        self.ax2 = self.fig.add_subplot(132)
        # surogate model
        self.ax3 = self.fig.add_subplot(133, projection='3d')
        plt.tight_layout()
        self.figNum = self.fig.number
        self.first_run = True
        self.ctr = 0

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
         

    def update_plot(self, X_sample, Y_sample):
        #check if plot exists or has been closed out.
        # return if closed so the program keeps running
        if plt.fignum_exists(self.figNum) == False:
            return
        
        # if self.scatter is None:
        #clear axes for redraw
        #self.ax1.clear() #don't clear this. the points should accumulate and function is static
        self.ax2.clear()
        self.ax3.clear()

        # create mesh and predict
        X, ei, mu, sigma = self.predict_plot_mesh()

        # reshape
        X0 = X[:,0].reshape(self.mesh_sample_dim, self.mesh_sample_dim)
        X1 = X[:,1].reshape(self.mesh_sample_dim, self.mesh_sample_dim)
        ei = ei.reshape(self.mesh_sample_dim, self.mesh_sample_dim)
        Y_mu_plot = mu.reshape(self.mesh_sample_dim, self.mesh_sample_dim)

        
        # OBJECTIVE FUNCTION AND SAMPLE PLOT
        if self.first_run == True:
            # to save time, just draw the static plot when this is true
            X_plot = X 
            #The objective functions for this suite are set up to only take 1 point at a time
            plot_FVals = [] # get ground truth from objective func
            for i in range(0, len(X_plot)):
                newFVals, noError = self.func_F(X_plot[i])
                if noError == False:
                    print("ERROR in objc func call update for loop")
                plot_FVals.append(newFVals)
            Y_plot = np.array(plot_FVals).reshape(self.mesh_sample_dim, self.mesh_sample_dim)
            
            #after this works. do not change
            self.ax1.plot_surface(X0, X1, Y_plot, cmap='viridis', alpha=0.7)
            self.first_run = False
        # The title gets updated every time to track the iterations
        self.ax1.set_title("Objective Function & " + str(len(Y_sample)) + " Samples")
        self.ax1.scatter(X_sample[:, 0], X_sample[:, 1], Y_sample, c='r', s=50)
 
        # ACQUISITION FUNCTION PLOTS
        self.ax2.contourf(X0, X1, ei, cmap='viridis')
        self.ax2.scatter(X_sample[:, 0], X_sample[:, 1], c='r', s=50)
        self.ax2.set_title('Acquisition Function (Expected Improvement)')

        # SURROGATE MODEL PLOT
        self.ax3.plot_surface(X0, X1, Y_mu_plot, cmap='viridis', alpha=0.7)
        self.ax3.scatter(X_sample[:, 0], X_sample[:, 1], Y_sample, c='r', s=50)
        self.ax3.set_title('Surrogate Model (GP Mean), Fitted to Samples')  


        plt.draw()
        plt.pause(0.0001)  # Pause to update the plot
        if self.ctr == 0:
            time.sleep(3)
        self.ctr = self.ctr + 1


    def fit_model(self, x, y):
        # call out to parent class to use surrogate model
        self.gp.fit(x,y)
        

    def model_predict(self, x):
        # call out to parent class to use surrogate model
        mu, sigma = self.gp.predict(x)
        return mu, sigma


    def predict_plot_mesh(self):
        # rather than creating X0 and X1, this dynamically creates the linspace for the mesh
        lbound = np.array(self.lbound[0])
        ubound = np.array(self.ubound[0])
    
        x_dims = []
        for i in range(0, len(lbound)):
            # Define range and step size
            x = np.linspace(lbound[i], ubound[i], self.mesh_sample_dim)
            x_dims.append(x) #the X0 and X1

        # Make into grid
        X = np.array(np.meshgrid(*x_dims)).T.reshape(-1, len(lbound)) #2 = number of dims in
        mu, sigma = self.model_predict(X)
        ei = self.bayesOptimizer.expected_improvement(X)

        return X, ei, mu, sigma


    def run(self):


        # set up the initial sample points  (randomly generated in this example)
        self.bayesOptimizer.initialize_starting_points(2)
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

            X_sample, Y_sample = self.bayesOptimizer.get_sample_points()
            self.update_plot(X_sample, Y_sample) #update matplot

        print("Optimized Solution")
        print(self.bayesOptimizer.get_optimized_soln())
        print("Optimized Outputs")
        print(self.bayesOptimizer.get_optimized_outs())


if __name__ == "__main__":
    tg = TestGraph()
    tg.run()
