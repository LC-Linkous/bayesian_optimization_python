#! /usr/bin/python3

##--------------------------------------------------------------------\
#   bayesian_optimization_python
#   './bayesian_optimization_python/src/bayesian_optimizer.py'
#   
#   Class for bayesian optimizer. Controlled by a driving test class as
#   a parent, which also passes arguments to the surrogate model.
#      
#
#   Author(s): Lauren Linkous
#   Last update: December 5, 2024
##--------------------------------------------------------------------\


import numpy as np
from numpy.random import Generator, MT19937
import sys


class BayesianOptimization:
    # arguments should take form: 
    # optimizer([[float, float, ...]], [[float, float, ...]], [[float, ...]], float, int,
    # func, func,
    # dataFrame,
    # class obj) 
    #  
    # opt_df contains class-specific tuning parameters
    # NO_OF_PARTICLES: int
    # weights: [[float, float, float]]
    # boundary: int. 1 = random, 2 = reflecting, 3 = absorbing,   4 = invisible
    # vlimit: float
    # 
   
    def __init__(self, lbound, ubound, targets,E_TOL, maxit,
                 obj_func, constr_func, 
                 opt_df,
                 parent=None):
        
        # Optional parent class func call to write out values that trigger constraint issues
        self.parent = parent 

        #unpack the opt_df standardized vals
        init_points = int(opt_df['INIT_PTS'][0])
        n_restarts = int(opt_df['NUM_RESTARTS'][0])
        xi = float(opt_df['XI'][0])


        # problem height and width
        heightl = np.shape(lbound)[0]
        widthl = np.shape(lbound)[1]
        heightu = np.shape(ubound)[0]
        widthu = np.shape(ubound)[1]

        # extract from array
        lbound = np.array(lbound[0])
        ubound = np.array(ubound[0])

        self.rng = Generator(MT19937())

        if ((heightl > 1) and (widthl > 1)) \
           or ((heightu > 1) and (widthu > 1)) \
           or (heightu != heightl) \
           or (widthl != widthu):
            
            if self.parent == None:
                pass
            else:
                self.parent.record_params()
                self.parent.debug_message_printout("ERROR: lbound and ubound must be 1xN-dimensional \
                                                        arrays  with the same length")
           
        else:

            
            if heightl == 1:
                lbound = lbound
        
            if heightu == 1:
                ubound = ubound

            self.lbound = lbound
            self.ubound = ubound


            '''
            self.M                      : An array of current X sample locations. 
            self.output_size            : An integer value for the output size of obj func
            self.Gb                     : Global best position, initialized with a large value.
            self.F_Gb                   : Fitness value corresponding to the global best position.
            self.F_Pb                   : Fitness value corresponding to the personal best position for each sample. Y sample locations. 
            self.targets                : Target values for the optimization process.
            self.maxit                  : Maximum number of iterations.
            self.E_TOL                  : Error tolerance.
            self.obj_func               : Objective function to be optimized.      
            self.constr_func            : Constraint function. Not used because of how points are generated.  
            self.iter                   : Current iteration count.
            self.allow_update           : Flag indicating whether to allow updates.
            self.Flist                  : List to store fitness values.
            self.Fvals                  : List to store fitness values.
            self.init_points            : Integer. Number of initial points before optimization.
            self.xi                     : Float. Encourages exploration in expected improvement.
            self.n_restarts             : Integer. Number of randomly genreated proposed sample candiates. 
            self.new_point              : Newest proposed/passed in point.
            '''

            self.M = []
            self.output_size = len(targets)
            self.Gb = sys.maxsize*np.ones((1,np.max([heightl, widthl])))   
            self.F_Gb = sys.maxsize*np.ones((1,self.output_size))
            self.F_Pb = []  
            self.targets = np.array(targets).reshape(-1, 1)         
            self.maxit = maxit                       
            self.E_TOL = E_TOL
            self.obj_func = obj_func                                             
            self.constr_func = constr_func   
            self.iter = 0
            self.allow_update = 0                                           
            self.Flist = []                                                 
            self.Fvals = []
            self.init_points = init_points    
            self.xi = xi
            self.n_restarts = n_restarts
            self.new_point = []
    

            # state machine control flow
            self.firstOptimizationRun = False
            self.doneInitializationBoolean = False
            self.ctr  = 0
            self.objective_function_case = 0 #initial pts by default
        

    # SURROGATE MODEL CALLS
    def fit_model(self, x, y):
        # call out to parent class to use surrogate model
        self.parent.fit_model(x,y)

    def model_predict(self, x) : #, outvar):
        # call out to parent class to use surrogate model
        mu, noError = self.parent.model_predict(x)
        return mu, noError
    
    def model_get_variance(self):
        sigma = self.parent.model_get_variance()
        return sigma

    # COMPLETION CHECKS
    def converged(self):
        # check if converged
        if (len(self.F_Gb) < 1): # no points, no sample
            return False
        convergence = np.linalg.norm(self.F_Gb) < self.E_TOL
        return convergence

    def maxed(self):
        # check if search max iterations hit
        max_iter = self.iter > self.maxit
        return max_iter

    def complete(self):
        done = self.converged() or self.maxed()
        return done

    # CHECK PROGRESS
    def check_global_local(self, Flist):
        # use L2 norm to check if fitness val is less than global best fitness
        # if yes, update with the new best point
        if (len(Flist) < 1) or (len(self.F_Gb)<1): # list is empty. not enought points yet
            return
        # if the current fitness is better than the current global best, replace the var.
        if np.linalg.norm(Flist) < np.linalg.norm(self.F_Gb):
            self.F_Gb = Flist
            self.Gb = 1*np.vstack(np.array(self.new_point))     

    def get_convergence_data(self):
        best_eval = np.linalg.norm(self.F_Gb)
        iteration = 1*self.iter
        return iteration, best_eval

    # GETTERS
    def get_optimized_soln(self):
        return self.Gb 
    
    def get_optimized_outs(self):
        return self.F_Gb

    def get_sample_points(self):
        # returns sample locations (self.M) and the associated evaluations (self.F_Pb)
        return self.M, self.F_Pb
   
    # OPTIMIZER FUNCTIONS
    def expected_improvement(self, X):
        X = np.atleast_2d(X)

        mu, noError = self.model_predict(X) #mean and standard deviation
        sigma = self.model_get_variance()
        mu_sample, _ = self.model_predict(self.M) #predict using sampled locations
        mu_sample_opt = np.min(mu_sample)
        
        imp = mu_sample_opt - mu - self.xi

        #Standardize Improvement
        #Z = np.where(sigma != 0, imp / sigma, 0)
        Z = np.divide(imp, sigma, out=np.zeros_like(imp), where=sigma!=0)
        ei = imp * (1 + np.tanh(Z))
        # # This introduces more stability into the model, 
        # # but the tested problems do much worse with exploration
        # for idx in range(0, len(sigma)):
        #     if sigma[idx] == 0.0:
        #         ei[:,idx] = np.multiply(ei[:,idx],0.0)
        
        return ei

    def propose_location(self):
        dim = len(self.lbound)
        min_val = 1
        min_x = None # may stay none if it doesn't minimize

        for x0 in self.rng.uniform(self.lbound.reshape(1,-1)[0], self.ubound.reshape(1,-1)[0], size=(self.n_restarts, dim)):
            x, f_x = self.minimize(x0)
            #using the l2norm to handle single and multi objective
            if np.linalg.norm(f_x) < np.linalg.norm(min_val):
                min_val = f_x
                min_x = x

        if min_x is None:
            # Return a random point in bounds
            min_x = self.rng.uniform(self.lbound.reshape(1, -1)[0], self.ubound.reshape(1, -1)[0])


        return np.array([min_x])


    def minimize(self, x0, max_iter=100, tol=1e-6):
        x = np.array(x0)
        alpha = 0.1
        for i in range(max_iter):
            grad = self.numerical_gradient(x, tol)
            x_new = np.clip(x - alpha * grad, self.lbound.reshape(1,-1)[0], self.ubound.reshape(1,-1)[0])
            if np.linalg.norm(x_new - x) < tol:
                break
            x = x_new
        f_x = -self.expected_improvement(np.array([x]))
        return x, f_x

    def numerical_gradient(self, x, eps=1e-8):
        grad = np.zeros_like(x)
        for i in range(len(x)):
            x_eps = x.copy()
            x_eps[i] += eps
            ei_x_eps = -self.expected_improvement(np.array([x_eps]))
            ei_x = -self.expected_improvement(np.array([x]))
            grad_arr = (ei_x_eps - ei_x) / eps
            grad[i] = np.linalg.norm(grad_arr) # Aggregate. explore other methods later. (summation?)

        return grad

    def error_message_generator(self, msg):
        # for error messages, but also repurposed for general updates
        if self.parent == None:
            print(msg)
        else:
            self.parent.debug_message_printout(msg)

    def call_objective(self, allow_update=False):

       # may have re-run in an upstream optimizer
        # so this may be triggered when the point should not be sampled
        if allow_update == False:
            self.allow_update = 0
            return
        else:        
            self.allow_update = 1
        
        # case 0: first point of initial points (must have minimum 1)
        if self.objective_function_case == 0:
            new_M = self.rng.uniform(self.lbound.reshape(1,-1)[0], self.ubound.reshape(1,-1)[0], (1, len(self.ubound))).reshape(1,len(self.ubound))
            newFVals, noError = self.obj_func(new_M[0], self.output_size)  # Cumulative Fvals

            if noError == True:
                newFVals = np.array([newFVals])#.reshape(-1, 1)  # kept for comparison between other optimzers
                self.M = new_M
                self.F_Pb = newFVals
            else:
                msg = "ERROR: initial point creation issue"
                self.error_message_generator(msg)

            self.is_fitted = False

        # case 1: any other initial points before optimiation begins
        elif self.objective_function_case == 1:
            new_M = self.rng.uniform(self.lbound.reshape(1,-1)[0], self.ubound.reshape(1,-1)[0], (1, len(self.ubound))).reshape(1,len(self.ubound))
            newFVals, noError = self.obj_func(new_M[0], self.output_size) 
            if noError == True:
                newFVals = np.array([newFVals])#.reshape(-1, 1) 
                self.M = np.vstack([self.M, new_M])
                self.F_Pb = np.vstack((self.F_Pb, newFVals))#.reshape(-1, 1) 

            else:
                msg = "ERROR: objective function error when initilaizing random points"
                self.error_message_generator(msg)

            self.is_fitted = False

        # case 2: normal objective function calls, optimization running live
        elif self.objective_function_case == 2:
            if len(self.new_point) < 1: # point not set
                msg = "WARNING: no point set for objective call. Skipping. Ignore unless this is persistent."
                self.error_message_generator(msg)
                return

            newFVals, noError = self.obj_func(self.new_point[0], self.output_size)
            if noError==True:
                self.Fvals = np.array([newFVals])#.reshape(-1, 1) 
                self.Flist = abs(self.targets - self.Fvals)
                self.M = np.vstack((self.M, self.new_point))
                self.F_Pb = np.vstack((self.F_Pb, newFVals))#.reshape(-1, 1) 
                self.fit_model(self.M, self.F_Pb)

        else:
            msg = "ERROR: in call objective objective function evaluation"
            self.error_message_generator(msg)


        self.iter = self.iter + 1

    def step(self, suppress_output=False):

        if not suppress_output:
            msg = "\n-----------------------------\n" + \
                "STEP #" + str(self.iter) +"\n" + \
                "-----------------------------\n" + \
                "Completed Initial Sample #" + str(self.ctr) + " of " + str(self.init_points) + "\n" +\
                "Last Proposed Point:\n" + \
                str(self.new_point) +"\n" + \
                "Best Fitness Solution: \n" +\
                str(np.linalg.norm(self.F_Gb)) +"\n" +\
                "Best Particle Position: \n" +\
                str(np.hstack(self.Gb)) + "\n" +\
                "-----------------------------"
            self.error_message_generator(msg)

        if self.allow_update:      

            #initialize the first couple runs
            if self.doneInitializationBoolean == False:
                #1) the initial points need to be run
                #2) getter for sample points (to ensure standard formatting)
                #3) fit the surrogate model

                # running the initial point collection 
                if self.ctr < self.init_points:
                    
                    # these are split into 2 cases for easier debug
                    if self.ctr == 0: # first point
                        #calling the objective function with the first random pt 
                        self.objective_function_case = 0
                        
                    else: # rest of the points
                        self.objective_function_case = 1
                    self.ctr = self.ctr + 1
                
                else: #if self.ctr >= self.init_points: 
                    msg = "Model initialized with " + str(self.ctr) + " points. \n \
                    The iteration counter will start from " + str(self.iter) + "\n\n"
                    self.error_message_generator(msg)
                    # get the sample points out (to ensure standard formatting)
                    x_sample, y_sample = self.get_sample_points()
                    # fit GP model.
                    self.parent.fit_model(x_sample, y_sample)
                    ### ABOVE HERE IS THE END OF THE FULL INITIALIZATION SETUP###
                    self.doneInitializationBoolean = True
                    

            # NOT an elif, so that this condition is hit immediately after the setup is done above
            if self.doneInitializationBoolean == True:
                # set to case 2 for objective function call
                self.objective_function_case = 2

                # check if points are better than last global bests
                self.check_global_local(self.Flist)

                #self.sample_next_point() in original example, but now split up
                self.new_point = self.propose_location()

                # check if points are better than last global bests
                self.check_global_local(self.Flist)

                #self.sample_next_point(), but split up
                self.new_point = self.propose_location()
            

            # There is no handling boundaries for points in the bayesian optimizer
            # because the proposed location is already bounded by the problem space

            if self.complete() and not suppress_output:
                msg =  "\nPoints: \n" + str(self.Gb) + "\n" + \
                    "Iterations: \n" + str(self.iter) + "\n" + \
                    "Flist: \n" + str(self.F_Gb) + "\n" + \
                    "Norm Flist: \n" + str(np.linalg.norm(self.F_Gb)) + "\n"
                self.error_message_generator(msg)
