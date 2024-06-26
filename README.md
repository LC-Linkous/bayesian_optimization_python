# bayesian_optimization_python
Repository for Bayesian Optimization with a Gaussian Process surrogate model in Python

The class format is based off of the [adaptive timestep PSO optimizer](https://github.com/jonathan46000/pso_python) by [jonathan46000](https://github.com/jonathan46000) for data collection baseline. This repo does not feature any PSO optimization. Instead, the format has been used to retain modularity with other optimizers.

Now featuring AntennaCAT hooks for GUI integration and user input handling.

## Table of Contents
* [Bayesian Search](#bayesian-search)
* [Requirements](#requirements)
* [Implementation](#implementation)
    * [Constraint Handling](#constraint-handling)
    * [Multi-Objective Optimization](#multi-objective-optimization)
    * [Objective Function Handling](#objective-function-handling)
      * [Internal Objective Function Example](internal-objective-function-example)
* [Example Implementations](#example-implementations)
    * [Basic Sweep Example](#basic-sweep-example)
    * [Realtime Graph](#realtime-graph)
* [References](#references)
* [Publications and Integration](#publications-and-integration)
* [Licensing](#licensing)  

## Bayesian Search

Bayesian search, or Bayesian optimization, uses probabilistic models to efficiently optimize functions that are expensive to evaluate. It iteratively updates a Bayesian model of the objective function based on sampled evaluations, balancing exploration of uncertain regions with exploitation of promising areas. This approach is particularly effective in scenarios like hyperparameter tuning and experimental design where each evaluation is resource-intensive or time-consuming.

## Requirements

This project requires numpy and matplotlib. 

Use 'pip install -r requirements.txt' to install the following dependencies:

```python
contourpy==1.2.1
cycler==0.12.1
fonttools==4.51.0
importlib_resources==6.4.0
kiwisolver==1.4.5
matplotlib==3.8.4
numpy==1.26.4
packaging==24.0
pillow==10.3.0
pyparsing==3.1.2
python-dateutil==2.9.0.post0
six==1.16.0
zipp==3.18.1

```

## Implementation
### Constraint Handling
In this example, proposed points are generated within the provided bounds of the problem, so there is no extra constraint handling added. 


### Multi-Objective Optimization
The no preference method of multi-objective optimization, but a Pareto Front is not calculated. Instead, the best choice (smallest norm of output vectors) is listed as the output.

### Objective Function Handling
The optimizer minimizes the absolute value of the difference from the target outputs and the evaluated outputs. Future versions may include options for function minimization absent target values. 


#### Internal Objective Function Example

There are three functions included in the repository:
1) Himmelblau's function, which takes 2 inputs and has 1 output
2) A multi-objective function with 3 inputs and 2 outputs (see lundquist_3_var)
3) A single-objective function with 1 input and 1 output (see one_dim_x_test)

Each function has four files in a directory:
   1) configs_F.py - contains imports for the objective function and constraints, CONSTANT assignments for functions and labeling, boundary ranges, the number of input variables, the number of output values, and the target values for the output
   2) constr_F.py - contains a function with the problem constraints, both for the function and for error handling in the case of under/overflow. 
   3) func_F.py - contains a function with the objective function.
   4) graph.py - contains a script to graph the function for visualization.



<p align="center">
        <img src="https://github.com/LC-Linkous/bayesian_optimization_python/blob/main/media/himmelblau_plots.png" alt="Himmelblau function" height="250">
</p>
   <p align="center">Plotted Himmelblau Function with 3D Plot on the Left, and a 2D Contour on the Right</p>

```math
f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
```

| Global Minima | Boundary | Constraints |
|----------|----------|----------|
| f(3, 2) = 0                 | $-5 \leq x,y \leq 5$  |   | 
| f(-2.805118, 3.121212) = 0  | $-5 \leq x,y \leq 5$  |   | 
| f(-3.779310, -3.283186) = 0 | $-5 \leq x,y \leq 5$  |   | 
| f(3.584428, -1.848126) = 0  | $-5 \leq x,y \leq 5$   |   | 




<p align="center">
        <img src="https://github.com/LC-Linkous/bayesian_optimization_python/blob/main/media/lundquist_3var_plots.png" alt="Function Feasible Decision Space and Objective Space with Pareto Front" height="200">
</p>
   <p align="center">Plotted Multi-Objective Function Feasible Decision Space and Objective Space with Pareto Front</p>

```math
\text{minimize}: 
\begin{cases}
f_{1}(\mathbf{x}) = (x_1-0.5)^2 + (x_2-0.1)^2 \\
f_{2}(\mathbf{x}) = (x_3-0.2)^4
\end{cases}
```

| Num. Input Variables| Boundary | Constraints |
|----------|----------|----------|
| 3      | $0.21\leq x_1\leq 1$ <br> $0\leq x_2\leq 1$ <br> $0.1 \leq x_3\leq 0.5$  | $x_3\gt \frac{x_1}{2}$ or $x_3\lt 0.1$| 



<p align="center">
        <img src="https://github.com/LC-Linkous/bayesian_optimization_python/blob/main/media/1D_test_plots.png" alt="Function Feasible Decision Space and Objective Space with Pareto Front" height="200">
</p>
   <p align="center">Plotted Single Input, Single-objective Function Feasible Decision Space and Objective Space with Pareto Front</p>

```math
f(\mathbf{x}) = sin(5 * x^3) + cos(5 * x) * (1 - tanh(x^2))
```





## Example Implementations

### Basic Example
main_test.py provides a sample use case of the optimizer with tunable parameters.

### Realtime Graph

<p align="center">
        <img src="https://github.com/LC-Linkous/bayesian_optimization_python/blob/main/media/surrogate_model_progress.gif" alt="gif of surrogate model development through iterations" height="325">
</p>
<p align="center">Bayesian Optimization with a Gaussian Proccess Using Himmelblau. Left to Right: Objective function ground truth, areas of interest to the optimizer, and surrogate model development process.</p>

<br>
<br>
<p align="center">
        <img src="https://github.com/LC-Linkous/bayesian_optimization_python/blob/main/media/surrogate_model_1X.gif" alt="gif of surrogate model development through iterations" height="325">
</p>
<p align="center">Bayesian Optimization with a Gaussian Proccess Using Single Input, Single Objective Function. Left, Gaussian Process Regression Model. Right: Aquisition Function.</p>

<br>
<br>

<p align="center">
        <img src="https://github.com/LC-Linkous/bayesian_optimization_python/blob/main/media/surrogate_model_multi_obj.gif" alt="gif of surrogate model development through iterations" height="325">
</p>
<p align="center">Bayesian Optimization with a Gaussian Proccess Using Multi-Objective Function. Left, Objective Function Output and Sampling. Right, Surrogate Model GP Mean Fitting and Sample Points.</p>

<br>
<br>

main_test_graph.py provides an example using a parent class, and the self.suppress_output and detailedWarnings flags to control error messages that are passed back to the parent class to be printed with a timestamp. Additionally, a realtime graph shows particle locations at every step.

The figure above shows a low-resolution version of the optimization for example. In this first figure, the Far left plot is the objective ground truth function with sample locations in red. The center plot is the expected improvement, which highlights areas of interest to the optimizer. The far right plot is the current shape of the surrogate model, with sampled points from the ground truth in red. The other two graphs present a similar process for different dimensions of objective functions.


NOTE: if you close the graph as the code is running, the code will continue to run, but the graph will not re-open.

## References

This repo does not currently reference any code from papers for the Bayesian or Gaussian classes.

For the original code base, see the [adaptive timestep PSO optimizer](https://github.com/jonathan46000/pso_python) by [jonathan46000](https://github.com/jonathan46000)

## Publications and Integration
This software works as a stand-alone implementation, and as one of the optimizers integrated into AntennaCAT.

Publications featuring the code in this repo will be added as they become public.


## Licensing

The code in this repository has been released under GPL-2.0


