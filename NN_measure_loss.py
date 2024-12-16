import matplotlib.pyplot as plt
import numpy as np
from geomloss import SamplesLoss
import torch.nn as nn
import torch
from torch import optim
from sklearn.preprocessing import MaxAbsScaler
import pickle

############################# lorenz system
def lorenz(xyz, *, s=10, r=28, b=2.667):
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])

################################ experiment parameters
dt = 0.001 #time-step
num_steps = int(1e6) #length of trajectory simulation
y = np.zeros((num_steps + 1, 3))  
y[0] = np.array([-5.065457, -7.56735 , 19.060379])  # Set initial values
ts = []
ts.append(0)
tau = 100 #time-delay
method = 'delay_IM' #select either 'delay_IM' or 'IM' to choose between a delay-coordinate invariant measure loss or a state-coordinate invariant measure loss
num_samples = 2000 #
Nsteps = 30000
plot_every = 500

###############################################

for i in range(num_steps):
    ts.append((i+1)*dt)
    y[i + 1] = y[i] + lorenz(y[i]) * dt

noise_scale = 1
y_full = y+np.random.normal(0,noise_scale,(len(y),3))
y = y_full
transformer = MaxAbsScaler().fit(y)
y = transformer.transform(y)
ts = np.array(ts)
ts = torch.tensor(ts,dtype = torch.float,requires_grad = True)

def delay(X):
    dim = 6
    new = np.zeros((len(X)-tau*dim,dim))
    for i in range(dim):
        print(dim*tau-(i+1)*tau)
        new[:,i] = X[dim*tau-(i+1)*tau:-(1+i)*tau]
    return new

y_delay = delay(y[:,0])
Ty_true = y[tau:tau+len(y_delay)]
y = y[:len(y_delay)]
############################# Construct batches

import random
batch_ixs = list(range(0,len(y)))
ixs = random.sample(batch_ixs,num_samples)

############################## Build network 
# torch.manual_seed(123897612)
torch.manual_seed(12932)
net = nn.Sequential(
    nn.Linear(3, 100),
    nn.Tanh(),
    nn.Linear(100,100),
    nn.Tanh(),
    nn.Linear(100,100),
    nn.Tanh(),
    nn.Linear(100,3))

############################## Define Loss
loss = SamplesLoss(loss="energy")
optimizer = optim.Adam(net.parameters(), lr=1e-3)  
##############################  training loop

net.train()

losses = []
for step in range(Nsteps):
    y_batch = torch.tensor(y[ixs],dtype = torch.float,requires_grad = True)    
    Ty_batch = torch.tensor(Ty_true[ixs],dtype = torch.float,requires_grad = True) 
    y_delay_batch = torch.tensor(y_delay[ixs],dtype = torch.float,requires_grad = True)
    optimizer.zero_grad()      
    Ty = net(y_batch)
    TTy = net(Ty)
    TTTy = net(TTy)
    TTTTy = net(TTTy)
    TTTTTy = net(TTTTy)

    DIM = torch.cat((TTTTTy[:,0].unsqueeze(0),TTTTy[:,0].unsqueeze(0),TTTy[:,0].unsqueeze(0),TTy[:,0].unsqueeze(0),Ty[:,0].unsqueeze(0),y_batch[:,0].unsqueeze(0)),dim = 0).T
    #delay invariant measure
    if method =='delay_IM':
        L = loss(Ty,Ty_batch) + loss(DIM,y_delay_batch)
    if method == 'IM':
        L = loss(Ty,Ty_batch)
    losses.append(L.detach().numpy())
    # L = loss(Ty,y[1:])
    # L = torch.mean((Ty-y[1:])**2)
    
    print('iteration ', step, 'loss ', L)
    L.backward()
    optimizer.step()
    if step%plot_every == 0:
        # state coordinate push forward
        plt.scatter(Ty.detach().numpy()[:,0],Ty.detach().numpy()[:,1],s =1)
        plt.scatter(Ty_batch.detach().numpy()[:,0],Ty_batch.detach().numpy()[:,1], s =1)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.show()
        #delay coordinate push forward
        plt.scatter(DIM.detach().numpy()[:,0],DIM.detach().numpy()[:,1],s =1)
        plt.scatter(y_delay_batch.detach().numpy()[:,0],y_delay_batch.detach().numpy()[:,1], s =1)
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.show()
        #traj simulation
        x = y_batch[0]
        xs = []
        for i in range(int(1e4)):
            xs.append(x.detach().numpy())
            x  = net(x)
        xs = transformer.inverse_transform(np.array(xs))
        plt.scatter(xs[:,0],xs[:,2],s = .1)
        plt.show()
        
net.eval()
x = torch.tensor(y[0],dtype = torch.float)
xs = []

for i in range(int(1e4)):
    xs.append(x.detach().numpy())
    x  = net(x)
xs = transformer.inverse_transform(np.array(xs))
y = transformer.inverse_transform(y)
plt.scatter(xs[:,0],xs[:,2],s = .1)
plt.show() 
        
plt.plot(xs[:,0][:5000][5:],xs[:,2][:5000][5:],linewidth = .5,linestyle = '--',marker = 'o',markersize = 3) 

if method == 'delay_IM': 
    with open("DIM_recon.p", "wb") as f:
        pickle.dump([y,y[ixs],xs,losses], f)
if method == 'IM':
    with open("IM_recon.p", "wb") as f:
        pickle.dump([y,y[ixs],xs,losses], f)

plt.plot(losses)
plt.yscale('log')