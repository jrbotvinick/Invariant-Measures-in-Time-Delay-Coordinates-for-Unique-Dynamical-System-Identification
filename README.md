# Invariant-Measures-in-Time-Delay-Coordinates-for-Unique-Dynamical-System-Identification

This repository contains Python code based on the work *[Invariant Measures in Time-Delay Coordinates for Unique Dynamical System Idenfitification](https://arxiv.org/abs/2412.00589v1)* by J. Botvinick-Greenhouse, R. Martin, and Y. Yang (2025). 

In our paper, we present several theoretical results which justify the use of delay-coordinate invariant measures for performing data-driven system identification. We also present two computational algorithms for using delay-coordinate invariant measures to perform system identification in practice, shown below. In this repository, we present several numerical examples and tutorials which deploy these approaches across physical systems, including the Lorenz-63 equations, the Kuramoto--Sivashinsky equation, and partial observations of vortex shedding past a cylinder.


- `Tutorials`:
     - `tutorial1.ipynb`*: This notebook walks through an example implementation of Algorithm 1 (Trajectory based delay measure opt.) for performing data-driven system identification using delay-coordinate invariant measures.
     - `tutorial2.ipynb`: This notebook walks through an example implementation of Algorithm 2 (Pushforward based delay measure opt.), which is useful in situations when the parameter space is large and gradient based optimization is necessary. 

<div align="center">
<img width="600" alt="Screenshot 2025-06-17 at 12 27 43 PM" src="https://github.com/user-attachments/assets/e6006d1d-b4da-445b-b54d-a30f37fcbe49" />
</div>

- `Torus Rotation`
     - `torus_rotation.py`: Simulates long trajectories of the torus rotation for four different rotation numbers and plots the trajectory samples in both the 3D projected state-coordinates and a 3D delay coordinate system.

<div align="center">
<img width="600" alt="Screenshot 2025-06-17 at 12 28 36 PM" src="https://github.com/user-attachments/assets/373cb55c-deba-40ee-a0eb-22d5671e4cd2" />
</div>

 - `Lorenz Example`
     - `NN_measure_loss.py`: Learns dynamics from Lorenz-63 data using a neural network parameterization with delay-coordinate invariant measure and state-coordinate invariant measure loss functions.
     - `NN_measure_loss.py`: Repeats the learning 10 times and saves the simulation results from the learned models for error computations.
     - `compute_errors.py`: Computes the sliced Wasserstein distance between the neural network simulated trajectories and the ground truth attractor.
     - `plot_lorenz.py`: Plot the simulate trajectory for a single neural network training and visualzie both the state-coordinate and delay-coordinate pushforward measures. 

<div align="center">
<img width="600" alt="Screenshot 2025-06-17 at 12 29 09 PM" src="https://github.com/user-attachments/assets/30ea717b-9455-4b58-a4be-7b8597d55590" />
</div>

- `KS Example`
     - `KS_simulate.py`*: Generate simulations of the Kuramoto--Sivashinsky equation over a range of parameters and saves the data.
     - `KS_evaluate_landscape.py`: Uses the simulations to generate optimization landscapes for inferring the true parameter from partial data using both pointwise and delay measure objectives.
     - `KS_optimization.py`*: Uses the Nelder--Mead algorithm to perform optimization using the two objectives from a fixed initial condition.
     - `KS_optimization_rands.py`*: Repeats the optimization over 10 randomly chosen initial parameter guesses and computes the estimation errors.
     - `KS_plot.py`: Plots the full unobserved dynamics, observed data, optimization landscapes, and Nelder--Mead iterations.

<div align="center">
<img width="600" alt="Screenshot 2025-06-17 at 12 30 18 PM" src="https://github.com/user-attachments/assets/77560b93-e4db-425a-8c59-57bef1cf1649" />
</div>

- `Cylinder Flow Example`
     - `simulate_flow.py`*: Simulates the flow past cylinder at Reynold's number Re = 70 using the DQ29 LBM.
     - `flow_data_prep.py`: Converts lattice units to physical units and collects partially observed noisy data from velocity probes in fluid wake.
     - `learn_flow_push.py`: Learns the dynamics of the observables using a neural network parameterization and delay-coordinate invariant measure objective function.
     - `learn_flow_push.py`: Repeats the learning procedure for 10 different random network initializations and saves the results for error computations.
     - `compute_errors.py`: Computes the forecast errors for the neural network predicted trajectories on unseen testing data.
     - `plot_flow_sensors.py`: Visualizes the sensor locations in the fluid flow and plots the predicted trajectories.

<div align="center">
<img width="600" alt="Screenshot 2025-06-17 at 12 30 37 PM" src="https://github.com/user-attachments/assets/c4c031aa-48c1-4d60-adc9-9236e8c6840f" />
</div>



If you find our code helpful, please cite:

```
@article{botvinick2024invariant,
  title={Invariant Measures in Time-Delay Coordinates for Unique Dynamical System Identification},
  author={Botvinick-Greenhouse, Jonah and Martin, Robert and Yang, Yunan},
  journal={arXiv preprint arXiv:2412.00589},
  year={2024}
}
```

*Indicates that these files contain sections of code from [machine-learning-and-simulation](https://github.com/Ceyron/machine-learning-and-simulation?tab=MIT-1-ov-file) by Felix Köhler; see `Credits.md`.
