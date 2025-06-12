# This file contains modifications to code from https://github.com/Ceyron/machine-learning-and-simulation?tab=MIT-1-ov-file
# which is Copyright (c) 2021 Felix Köhler and licensed under the MIT License
# See the CREDITS.md file for full license information.


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pickle



num_steps = int(1e5)
dt = 0.1 #time-step

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
            print(self.dx,self.dt)
        
    
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
    print(mesh,DOMAIN_SIZE)
    
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
    plt.imshow(y.T[:,:5000],origin = 'lower',aspect = 'auto',cmap = 'jet')
    plt.show()
    
    return y


params = np.linspace(0.5,1.5,100).tolist()
params.append(1)
data = np.zeros((len(params),num_steps))

i = 0
for param in params:
    data[i] = experiment(param)[:,0]
    i+=1 
    print(param)


data_full = experiment(1)
with open("K.p", "wb") as f:
         pickle.dump([data,data_full], f)
         
         
    