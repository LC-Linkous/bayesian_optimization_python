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
#   Last update: August 18, 2024
##--------------------------------------------------------------------\


import numpy as np
import time
import matplotlib.pyplot as plt
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
#import himmelblau.configs_F as func_configs         # single objective, 2D input
import lundquist_3_var.configs_F as func_configs    # multi objective function


class TestGraph():
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

        # for handling multiple types of graphs
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
        PR_degree = 40
        # Polynomial Chaos Expansion vars
        PC_degree = 5 
        # KNN regression vars
        KNN_n_neighbors=3
        KNN_weights='uniform'  #options: 'uniform', 'distance'
        # Decision Tree Regression vars
        DTR_max_depth = 5  # options: ints


        #plotting vars - make sure plots and samples match
        self.mesh_sample_dim = 25
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
        

        # Matplotlib setup
        # # Initialize plot
        self.fig = plt.figure(figsize=(10, 7))

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
        
        if self.in_vars == 1:
            self.plot_1D(X_sample, Y_sample)
        elif self.in_vars == 2:
            if self.out_vars == 1: #single objective
                self.plot_2D_single(X_sample, Y_sample)
            else:
                print("ERROR: objective function not currently supported for plots")
        elif self.in_vars == 3:
            if self.out_vars == 2:
                self.plot_2D_multi(X_sample, Y_sample)
            else:
                print("ERROR: objective function not currently supported for plots")

        else:
            print("ERROR: objective function not currently supported for plots")


    def plot_1D(self, X_sample, Y_sample) :
        lbound = np.array(self.lbound[0])
        ubound = np.array(self.ubound[0])
    
        X = np.linspace(lbound, ubound, self.mesh_sample_dim).reshape(-1, 1)

        mu, noError = self.model_predict(X)
        sigma = self.model_get_variance()
        ei = self.bayesOptimizer.expected_improvement(X)

        # initialize the plot
        if self.first_run == True:
            self.ax1 = self.fig.add_subplot(121)
            self.ax2 = self.fig.add_subplot(122)

            plt.tight_layout()
            self.first_run = False

        # clear plot
        self.ax1.clear()
        self.ax2.clear()
    

        # OBJECTIVE FUNCTION, SAMPLES, AND PREDICTION
        #The objective functions for this suite are set up to only take 1 point at a time
        plot_FVals = [] # get ground truth from objective func
        for i in range(0, len(X)):
            newFVals, noError = self.func_F(X[i])
            if noError == False:
                print("ERROR in objc func call update for loop")
            plot_FVals.append(newFVals)
        Y_plot = np.array(plot_FVals).reshape(-1, 1)      

        self.ax1.plot(X, Y_plot, 'r:', label='Objective Function')
        self.ax1.plot(X_sample, Y_sample, 'r.', markersize=10, label='Samples')
        self.ax1.plot(X, mu, 'b-', label='GP Mean')
        self.ax1.fill_between(X.ravel(), (mu - 1.96 * sigma).ravel(), (mu + 1.96 * sigma).ravel(), alpha=0.2, color='b')
        self.ax1.set_title('Objective Function and Surrogate Model, Samples: ' + str(self.ctr))
        self.ax1.legend()
        
        # EXPECTED IMPROVEMENT
        self.ax2.plot(X, ei, 'g-', label='Expected Improvement')
        self.ax2.set_title('Acquisition Function')
        self.ax2.legend()
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

        if self.ctr == 0:
            time.sleep(5)
        self.ctr = self.ctr + 1

        # print statements for README update
        # if self.ctr == 1:
        #     plt.savefig("1d_bayes_save_1.png")
        # elif self.ctr == 2:
        #     plt.savefig("1d_bayes_save_2.png")           
        # elif self.ctr == 3:
        #     plt.savefig("1d_bayes_save_3.png")        
        # elif self.ctr == 4:
        #     plt.savefig("1d_bayes_save_4.png")      
        # elif self.ctr == 5:
        #     plt.savefig("1d_bayes_save_5.png")      
        # elif self.ctr == 10:
        #     plt.savefig("1d_bayes_save_10.png")      
        # elif self.ctr == 15:
        #     plt.savefig("1d_bayes_save_15.png")      
        # elif self.ctr == 20:
        #     plt.savefig("1d_bayes_save_20.png")      




    def plot_2D_single(self, X_sample, Y_sample):

        # create mesh and predict
        X, ei, mu, sigma = self.predict_plot_mesh()

        # reshape
        X0 = X[:,0].reshape(self.mesh_sample_dim, self.mesh_sample_dim)
        X1 = X[:,1].reshape(self.mesh_sample_dim, self.mesh_sample_dim)
        ei = ei.reshape(self.mesh_sample_dim, self.mesh_sample_dim)
        Y_mu_plot = mu.reshape(self.mesh_sample_dim, self.mesh_sample_dim)

        # Initialize plot
        if self.first_run == True:
            # objective function and sample points
            self.ax1 = self.fig.add_subplot(131, projection='3d')
            # aquisition function
            self.ax2 = self.fig.add_subplot(132)
            # surogate model (pareto front)
            self.ax3 = self.fig.add_subplot(133, projection='3d')
            plt.tight_layout()

            # OBJECTIVE FUNCTION AND SAMPLE PLOT
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

        # if self.scatter is None:
        #clear axes for redraw
        #self.ax1.clear() #don't clear this. the points should accumulate and function is static
        self.ax2.clear()
        self.ax3.clear()
      
        # # OBJECTIVE FUNCTION AND SAMPLE PLOT
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
        self.ax3.set_title('Surrogate Model, \n Fitted to Samples')  


        plt.draw()
        plt.pause(0.0001)  # Pause to update the plot
        if self.ctr == 0:
            time.sleep(3)
        self.ctr = self.ctr + 1

        if self.ctr == 1:
            plt.savefig("2d_bayes_save_1.png")
        elif self.ctr == 2:
            plt.savefig("2d_bayes_save_2.png")           
        elif self.ctr == 3:
            plt.savefig("2d_bayes_save_3.png")        
        elif self.ctr == 4:
            plt.savefig("2d_bayes_save_4.png")      
        elif self.ctr == 5:
            plt.savefig("2d_bayes_save_5.png")      
        elif self.ctr == 10:
            plt.savefig("2d_bayes_save_10.png")      
        elif self.ctr == 15:
            plt.savefig("2d_bayes_save_15.png")      
        elif self.ctr == 20:
            plt.savefig("2d_bayes_save_20.png")      
        elif self.ctr == 30:
            plt.savefig("2d_bayes_save_30.png")      
        elif self.ctr == 40:
            plt.savefig("2d_bayes_save_40.png")      
        elif self.ctr == 50:
            plt.savefig("2d_bayes_save_50.png")        
        elif self.ctr == 60:
            plt.savefig("2d_bayes_save_60.png")      
        elif self.ctr == 70:
            plt.savefig("2d_bayes_save_70.png")      
        elif self.ctr == 100:
            plt.savefig("2d_bayes_save_100.png")      
        elif self.ctr == 120:
            plt.savefig("2d_bayes_save_120.png")    
        elif self.ctr == 150:
            plt.savefig("2d_bayes_save_150.png")      
        elif self.ctr == 200:
            plt.savefig("2d_bayes_save_200.png")      



    def plot_2D_multi(self, X_sample, Y_sample):
        #create mesh and predict
        lbound = np.array(self.lbound[0])
        ubound = np.array(self.ubound[0])    
        X = np.linspace(lbound, ubound, self.mesh_sample_dim)#.reshape(-1, 1)

        # Initialize plot
        if self.first_run == True:
            # objective function pareto front
            self.ax1 = self.fig.add_subplot(121, projection='3d')
            # surrogate model pareto front
            self.ax2 = self.fig.add_subplot(122, projection='3d')
            plt.tight_layout()
            self.first_run = False
        
        # clear plots
        self.ax1.clear()
        self.ax2.clear()
      
        # # OBJECTIVE FUNCTION AND SAMPLE PLOT
        #The objective functions for this suite are set up to only take 1 point at a time
        plot_FVals = [] # get ground truth from objective func
        for i in range(0, len(X)):
            newFVals, noError = self.func_F(X[i])
            if noError == False:
                print("ERROR in objc func call update for loop")
            plot_FVals.append(newFVals)
        Y = np.array(plot_FVals).reshape(-1, 2)
        Y0 = Y[:,0].reshape(-1, 1)
        Y1 = Y[:,1].reshape(-1, 1)
        self.ax1.scatter(Y0, Y1, c='b', label='Objective Func Output')
        self.ax1.set_title("Objective Function Output Sampling \n & " + str(len(Y_sample)) + " Samples")
        self.ax1.scatter(Y_sample[:, 0], Y_sample[:, 1], c='r', marker="X", s=50, label='Samples')
        self.ax1.legend()
        
        # the 3 dimensional arrays are too much to put in at once, so do it in pieces.
        mu_arr = []
        ei_arr = []
        for x in X:
            mu, noError = self.model_predict(x)
            ei = self.bayesOptimizer.expected_improvement(x)
            mu_arr.append(mu)
            ei_arr.append(ei)

        # reshape
        ei = np.array(ei_arr).reshape(-1, 1)
        Y_mu_plot = np.array(mu_arr).reshape(-1, 2)

        # SURROGATE MODEL PLOT
        self.ax2.scatter(Y_mu_plot[:,0], Y_mu_plot[:,1], c='b', label='Surrogate Output Fitness')
        self.ax2.scatter(Y_sample[:, 0], Y_sample[:, 1], c='r', marker="X", s=50, label='Sample Fitness')
        self.ax2.set_title('Surrogate Model, \n Fitted to Samples')  
        self.ax2.legend()
        
        plt.draw()
        plt.pause(0.0001)  # Pause to update the plot
        if self.ctr == 0:
            time.sleep(3)
        self.ctr = self.ctr + 1

        if self.ctr == 1:
            plt.savefig("2d_mult_bayes_save_1.png")
        elif self.ctr == 2:
            plt.savefig("2d_mult_bayes_save_2.png")           
        elif self.ctr == 3:
            plt.savefig("2d_mult_bayes_save_3.png")        
        elif self.ctr == 4:
            plt.savefig("2d_mult_bayes_save_4.png")      
        elif self.ctr == 5:
            plt.savefig("2d_mult_bayes_save_5.png")      
        elif self.ctr == 10:
            plt.savefig("2d_mult_bayes_save_10.png")      
        elif self.ctr == 15:
            plt.savefig("2d_mult_bayes_save_15.png")      
        elif self.ctr == 20:
            plt.savefig("2d_mult_bayes_save_20.png")      
        elif self.ctr == 30:
            plt.savefig("2d_mult_bayes_save_30.png")      
        elif self.ctr == 40:
            plt.savefig("2d_mult_bayes_save_40.png")      
        elif self.ctr == 50:
            plt.savefig("2d_mult_bayes_save_50.png")        
        elif self.ctr == 60:
            plt.savefig("2d_mult_bayes_save_60.png")      
        elif self.ctr == 70:
            plt.savefig("2d_mult_bayes_save_70.png")      
        elif self.ctr == 100:
            plt.savefig("2d_mult_bayes_save_100.png")      
        elif self.ctr == 120:
            plt.savefig("2d_mult_bayes_save_120.png")    
        elif self.ctr == 150:
            plt.savefig("2d_mult_bayes_save_150.png")      
        elif self.ctr == 200:
            plt.savefig("2d_mult_bayes_save_200.png")      


 
  
    # SURROGATE MODEL FUNCS
    def fit_model(self, x, y):
        # call out to parent class to use surrogate model
        self.sm.fit(x,y)
        

    def model_predict(self, x):
        # call out to parent class to use surrogate model
        mu, noError = self.sm.predict(x, self.out_vars)
        return mu, noError

    def model_get_variance(self):
        variance = self.sm.calculate_variance()
        return variance

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
        mu, noError = self.model_predict(X)
        sigma = self.model_get_variance()
        ei = self.bayesOptimizer.expected_improvement(X)

        return X, ei, mu, sigma


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

            X_sample, Y_sample = self.bayesOptimizer.get_sample_points()
            self.update_plot(X_sample, Y_sample) #update matplot

        print("Optimized Solution")
        print(self.bayesOptimizer.get_optimized_soln())
        print("Optimized Outputs")
        print(self.bayesOptimizer.get_optimized_outs())

        time.sleep(30) #hold the graph open


if __name__ == "__main__":
    tg = TestGraph()
    tg.run()
