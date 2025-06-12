# This file contains modifications code from https://github.com/Ceyron/machine-learning-and-simulation?tab=MIT-1-ov-file
# which is Copyright (c) 2021 Felix Köhler and licensed under the MIT License
# See the CREDITS.md file for full license information.


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from geomloss import SamplesLoss
import torch.nn as nn
import torch
from torch import optim
import pickle
from torchdiffeq import odeint_adjoint as odeint, odeint_event
from sklearn.preprocessing import MaxAbsScaler
from ot.sliced import sliced_wasserstein_distance
from scipy.optimize import minimize



################################ experiment parameters
dt = 0.1 
freq = 30
num_steps = int(1e5) 
############################################### generate data

def experiment(a):
    DOMAIN_SIZE = 100.0
    N_DOF = 200
    
    class KuramotoSivashinsky():
        def __init__(
            self,
            L,
            N,
            dt,
        ):
            self.L = L
            self.N = N
            self.dt = dt
            self.dx = L / N
    
            wavenumbers = jnp.fft.rfftfreq(N, d=L / (N * 2 * jnp.pi))
            self.derivative_operator = 1j * wavenumbers
            
    
            linear_operator = - a* self.derivative_operator**2 - a* self.derivative_operator**4
            self.exp_term = jnp.exp(dt * linear_operator)
            self.coef = jnp.where(
                linear_operator == 0.0,
                dt,
                (self.exp_term - 1.0) / linear_operator,
            )
    
            self.alias_mask = (wavenumbers < 2/3 * jnp.max(wavenumbers))
        
        def __call__(
            self,
            u,
        ):
            u_nonlin = - 0.5 * u**2
            u_hat = jnp.fft.rfft(u)
            u_nonlin_hat = jnp.fft.rfft(u_nonlin)
            u_nonlin_hat = self.alias_mask * u_nonlin_hat
            u_nonlin_der_hat = self.derivative_operator * u_nonlin_hat
    
            u_next_hat = self.exp_term * u_hat + self.coef * u_nonlin_der_hat
            u_next = jnp.fft.irfft(u_next_hat, n=self.N)
            return u_next
        
    mesh = jnp.linspace(0.0, DOMAIN_SIZE, N_DOF, endpoint=False)
    
    def KS_trajectory():
        u_0 = jnp.sin( 2* jnp.pi * mesh / DOMAIN_SIZE)
        ks_stepper = KuramotoSivashinsky(
            L=DOMAIN_SIZE,
            N=N_DOF,
            dt=dt,
            )
        ks_stepper = jax.jit(ks_stepper)
        u_current = u_0
        trj = [u_current, ]
        for i in range(num_steps-1):
            u_current = ks_stepper(u_current)
            trj.append(u_current)
        
        trj = jnp.stack(trj)
        return trj
    
    
    y = KS_trajectory()
    y = np.asarray(y)
   # plt.imshow(y.T[:,:5000],origin = 'lower',aspect = 'auto',cmap = 'jet')
  #  plt.show()
    
    return y[:,0][::freq]

obs = experiment(1)
scale = 0.25
noise = np.random.normal(0,scale,len(obs))
obs = obs+noise
def delay(X,dim,tau):
    new = np.zeros((len(X)-tau*dim,dim))
    for i in range(dim):
        new[:,i] = X[dim*tau-(i+1)*tau:-(1+i)*tau]
    return new

def MLoss(a):
    s1 = obs
    s2 = experiment(a)
    dim,tau = 5, 1
    ds1 = delay(s1,dim,tau)
    ds2 = delay(s2,dim,tau)
    return sliced_wasserstein_distance(ds1,ds2)
    
def L2(a):
    s1 = obs
    s2 = experiment(a)
    return np.mean((s1-s2)**2)   


losses = []     
def callback(x):
    print(x)
    losses.append(x)
  
losses2 = []     
def callback2(x):
    print(x)
    losses2.append(x)
    
errors_point = []
errors_DIM = []
for i in range(10):
    init = np.random.uniform(0.5,1.5)
    res2 = minimize(L2, init, method='Nelder-Mead', bounds = [(0.5,1.5)],callback = callback2,options={'maxiter': 15, 'disp': True})
    res = minimize(MLoss, init, method='Nelder-Mead', bounds = [(0.5,1.5)],callback = callback,options={'maxiter': 15, 'disp': True})
    
    paramL2 = res2.x
    paramDIM = res.x
    print(paramL2)
    print(paramDIM)
    errors_point.append(np.abs(res2.x[0]-1))
    errors_DIM.append(np.abs(res.x[0]-1))
    print('Iteration: ', i)

print('Measure mean:', np.mean(errors_DIM), 'Measure STD:', np.std(errors_DIM))
print('Pointwise mean:', np.mean(errors_point), 'Pointwise STD:', np.std(errors_point))


    


    