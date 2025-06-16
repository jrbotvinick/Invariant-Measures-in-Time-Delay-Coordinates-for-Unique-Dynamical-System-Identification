# Invariant-Measures-in-Time-Delay-Coordinates-for-Unique-Dynamical-System-Identification

This repository contains Python code for reproducing the results in our work *[Invariant Measures in Time-Delay Coordinates for Unique Dynamical System Idenfitification](https://arxiv.org/abs/2412.00589v1).* 

- `Torus Rotation`

     - `torus_rotation.py`: Simulates long trajectories of the torus rotation for four different rotation numbers and plots the trajectory samples in both the 3D projected state-coordinates and a 3D delay coordinate syste.

 - `Lorenz Example`

     - `NN_measure_loss.py`: Learns dynamics from Lorenz-63 data using a neural network parameterization with delay-coordinate invariant measure and state-coordinate invariant measure loss functions.
     - `NN_measure_loss.py`: Repeats the experiments from `NN_measure_loss.py` 10 times and saves the simulation results from the learned models for error computations.
     - `compute_errors.py`: Computes the sliced Wasserstein distance between the neural network simulated trajectories and the ground truth attractor.
     - `plot_lorenz.py`: Plot the simulate trajectory for a single neural network training and visualzie both the state-coordinate and delay-coordinate pushforward measures. 

