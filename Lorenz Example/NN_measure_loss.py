import matplotlib.pyplot as plt
import numpy as np
from geomloss import SamplesLoss
import torch.nn as nn
import torch
from torch import optim
from sklearn.preprocessing import MaxAbsScaler
import pickle

############################# lorenz system; see https://en.wikipedia.org/wiki/Lorenz_system
def lorenz(xyz, *, s=10, r=28, b=2.667):
    x, y, z = xyz
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])

def lorenz_vec(xyz, *, s=10, r=28, b=2.667):
    x, y, z = xyz[:,0], xyz[:,1], xyz[:,2]
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return np.array([x_dot, y_dot, z_dot])


################################ experiment parameters
dt = 0.01 #time-step
num_steps = int(2e5) #length of trajectory simulation
y = np.zeros((num_steps + 1, 3))  
y[0] = np.array([-5.065457, -7.56735 , 19.060379])  # Set initial values
tau = 10 #time-delay
method = 'delay_IM' #select either 'delay_IM' or 'IM' to choose between a delay-coordinate invariant measure loss or a state-coordinate invariant measure loss
num_samples = 500 #the number of trajectory samples we want to learn from. The runtime is very sensitive to this number since we do not use minibatches. For > 2000, one should partition into mini-batches. 
Nsteps = 10000 #how many training iterations to take 
plot_every = 200 #how often to plot training result
dim = 5
###############################################

#simulate trajectory
for i in range(num_steps):
    y[i + 1] = y[i] + lorenz(y[i]) * dt

#rescale for NN learning
#
#
#time-delay map
def delay(X):
    new = np.zeros((len(X)-tau*dim,dim))
    for i in range(dim):
        print(dim*tau-(i+1)*tau)
        new[:,i] = X[dim*tau-(i+1)*tau:-(1+i)*tau]
    return new

y_delay_full = delay(y[:,0])

transformer = MaxAbsScaler().fit(y)
y = transformer.transform(y)
#form delayed trajectory
y_delay = delay(y[:,0])

y = y[:-(dim-1)*tau]
Ty_true = y[tau:]
y = y[:-tau]
#take random samples for training
import random
batch_ixs = list(range(0,len(y)))

############################## Build network 
torch.manual_seed(1252) #random seed so that initialization is controlled for comparison
np.random.seed(13531)


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
  


##############################  Training loop
net.train()
losses = []
for step in range(Nsteps):
    ixs = random.sample(batch_ixs,num_samples)

    y_batch = torch.tensor(y[ixs],dtype = torch.float,requires_grad = True)     
    Ty_batch = torch.tensor(Ty_true[ixs],dtype = torch.float,requires_grad = True)  
    y_delay_batch = torch.tensor(y_delay[ixs],dtype = torch.float,requires_grad = True) 


    optimizer.zero_grad()
    
    sols = []
    yy = y_batch
    for i in range(tau*dim):
        sols.append(yy)
        yy = net(yy)*dt+yy

    sol = torch.stack(sols, dim=0)

    Ty = sol[tau,:,:]
    DIM = torch.flip(sol[::tau,:,0].T,[1])
    
    #dChoose between state-coordinate or delay-coordinate invariant measure loss
    if method =='delay_IM':
        L = loss(Ty,Ty_batch) + loss(DIM,y_delay_batch)
    if method == 'IM':
        L = loss(Ty,Ty_batch)
        
        
    losses.append(L.detach().numpy())  
    print('iteration : ', step, '| loss :', L.detach().numpy())
    L.backward()
    optimizer.step()
    
    #plot current progress
    if step%plot_every == 0:
        # state coordinate push forward
        plt.title('State-Coordinate Pushforward Samples',fontsize = 15)
        plt.scatter(Ty_batch.detach().numpy()[:,0],Ty_batch.detach().numpy()[:,1], s =1,label = 'Data')
        plt.scatter(Ty.detach().numpy()[:,0],Ty.detach().numpy()[:,1],s =1, label = 'Model')
        plt.legend(loc = 'upper right')
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.show()
        #delay coordinate push forward
        plt.title('Delay-Coordinate Pushforward Samples',fontsize = 15)
        plt.scatter(y_delay_batch.detach().numpy()[:,0],y_delay_batch.detach().numpy()[:,1], s =1,label = 'Data')
        plt.scatter(DIM.detach().numpy()[:,0],DIM.detach().numpy()[:,1],s =1,label = 'Model')
        plt.legend(loc = 'upper right')
        plt.show()
        #plot a long simulated trajectory
        x = y_batch[0]
        xs = []
        for i in range(int(1e4)):
            xs.append(x.detach().numpy())
            x  = x+dt*net(x)
        xs = transformer.inverse_transform(np.array(xs))
        plt.title('Model Simulated Trajectory',fontsize = 15)
        plt.plot(xs[:,0],xs[:,2],linewidth = 1)
        plt.show()


#Simulate a long trajectory using the learned model        
net.eval()
x = torch.tensor(np.array([0.1,0.1,0.5]),dtype = torch.float)
xs = []

for i in range(num_steps):
    xs.append(x.detach().numpy())
    x  = net(x)*dt+x
    
xs = transformer.inverse_transform(np.array(xs))
plt.scatter(xs[:,0],xs[:,2],s = .1)
plt.show() 
        
plt.plot(xs[:,0][:5000][5:],xs[:,2][:5000][5:],linewidth = .5,linestyle = '--',marker = 'o',markersize = 3) 


#compute pushforwards 


sols = []
yy = torch.tensor(y,dtype = torch.float)
for i in range(tau*dim):
    sols.append(transformer.inverse_transform(yy.detach().numpy()))
    yy = net(yy)*dt+yy

sol = np.array(sols)

Ty = sol[tau,:,:]
DIM = np.flip(sol[::tau,:,0].T,axis = 1)

plt.scatter(Ty[:,0],Ty[:,1],s=0.01)
plt.show()


y = transformer.inverse_transform(y)
Ty_true = transformer.inverse_transform(Ty_true)
#save the data for plotting
if method == 'delay_IM': 
    with open("DIM_recon.p", "wb") as f:
        pickle.dump([y,Ty_true,y_delay_full,y[ixs],xs,Ty,DIM,losses], f)
if method == 'IM':
    with open("IM_recon.p", "wb") as f:
        pickle.dump([y,Ty_true,y_delay_full,y[ixs],xs,Ty,DIM,losses], f)
