# Invariant-Measures-in-Time-Delay-Coordinates-for-Unique-Dynamical-System-Identification

This repository contains Python code for reproducing the results in our work *[Invariant Measures in Time-Delay Coordinates for Unique Dynamical System Idenfitification](https://arxiv.org/abs/2412.00589v1).* 

- `Torus Rotation`

     - `torus_rotation.py`: Simulates long trajectories of the torus rotation for four different rotation numbers and plots the trajectory samples in both the 3D projected state-coordinates and a 3D delay coordinate syste.

 - `Lorenz Example`

     - `NN_measure_loss.py`: Learns dynamics from Lorenz-63 data using a neural network parameterization with delay-coordinate invariant measure and state-coordinate invariant measure loss functions.
     - `NN_measure_loss.py`: Repeats the experiments from `NN_measure_loss.py` 10 times and saves the simulation results from the learned models for error computations.
     - `compute_errors.py`: Computes the sliced Wasserstein distance between the neural network simulated trajectories and the ground truth attractor.
     - `plot_lorenz.py`: Plot the simulate trajectory for a single neural network training and visualzie both the state-coordinate and delay-coordinate pushforward measures. 

- `KS Example`

     - `KS_simulate.py`: Generate simulations of the Kuramoto--Sivashinsky equation over a range of parameters and saves the data.
     - `KS_evaluate_landscape.py`: Uses the simulations to generate optimization landscapes for inferring the true parameter from partial data using both pointwise and delay measure objectives.
     - `KS_optimization.py`: Uses the Nelder--Mead algorithm to perform optimization using the two objectives from a fixed initial condition.
     - `KS_optimization_rands.py`: Repeats the optimization over 10 randomly chosen initial parameter guesses and computes the estimation errors.
     - `KS_plot.py`: Plots the full unobserved dynamics, observed data, optimization landscapes, and Nelder--Mead iterations.

- `Cylinder Flow Example`

     - `simulate_flow.py`: Simulates the flow past cylinder at Reynold's number Re = 70 using the DQ29 LBM.*
     - `flow_data_prep.py`: Converts lattice units to physical units and collects partially observed noisy data from velocity probes in fluid wake.
     - `learn_flow_push.py`: Learns the dynamics of the observables using a neural network parameterization and delay-coordinate invariant measure objective function.
     - `learn_flow_push.py`: Repeats the learning procedure for 10 different random network initializations and saves the results for error computations.
     - `compute_errors.py`: Computes the forecast errors for the neural network predicted trajectories on unseen testing data.
     - `plot_flow_sensors.py`: Visualizes the sensor locations in the fluid flow and plots the predicted trajectories. 

     
