import matplotlib.pyplot as plt
import numpy as np
from geomloss import SamplesLoss
import torch.nn as nn
import torch
import pickle
from sklearn.preprocessing import StandardScaler as Scaler 

for iii in range(10):
    
    with open("flow_sensors.p", "rb") as f:
        data = pickle.load(f)
        
    coeffs,testing = data[0], data[1]
    dt_data = data[-4]
    data_sample_rate = data[3]
    
    ################################ experiment parameters
    n_obs,dim_state = np.shape(coeffs) #state dimension
    ode_sampling_freq =10 #how frequently the ode integrator is called
    dt = data[-2]/ode_sampling_freq #ode time-step
    tau = 2 #time-delay used for the quickly sampled trajectories
    method = 'P'  #Choice of loss function. Options are "P" for pointwise, "IM" for invariant measure, and "DIM" for delay invariant measure.
    loss_measures = 'energy' #loss function on the space of probability measures
    Nsteps = 10000 #number of training iterations to take 
    dim = 4 #embedding dimension
    visualize = True #wether to visualize while training
    plot_every = 500 #how often to plot results
    n_simulate = int(5e5) #trajectory simulation length after training
    ############################################### generate data
    
    def delay(Y,dim,tau):
        delayed = torch.zeros((dim_state,len(Y)-tau*(dim-1),dim))
        for i in range(dim):
            delayed[:,:,i] = Y[dim*tau-(i+1)*tau:][:len(Y)-tau*(dim-1)].T
        return delayed
    
    
    #coeffs = coeffs[:100]
    scaler = Scaler()
    scaler.fit(coeffs)
    y = torch.tensor( scaler.transform(coeffs),dtype = torch.float)
    Ty_true = y[tau:]
    y_delayed = delay(y,dim,tau)
    
    ################### define model
    nodes = 100
    
    sample_loss = SamplesLoss(loss=loss_measures,blur = 0.1)
    
    
    class FlowStep(nn.Module):
        def __init__(self, dt):
            super().__init__()
            self.L1 = nn.Linear(len(coeffs[0]), nodes,bias=True)
            self.T1 = nn.Tanh()
            self.L2 = nn.Linear(nodes, nodes,bias=True)
            self.T2 = nn.Tanh()
            self.L2 = nn.Linear(nodes, nodes,bias=True)
            self.T2 = nn.Tanh()
            self.L3 = nn.Linear(nodes, len(coeffs[0]),bias=True)
            self.dt = dt
            
        def forward(self, X):
            
            V = X.clone()
            V =  self.L1(V)
            V = self.T1(V)
            V = self.L2(V)
            V = self.T2(V)
            V = self.L3(V)
    
            return V
    
    
    T = FlowStep(dt)
    optimizer = torch.optim.Adam(T.parameters(),lr = 1e-3)
    params = np.zeros((Nsteps,4))
    
    
    
    for ii in range(Nsteps):   
            optimizer.zero_grad()
            sols = []
            yy = y.clone()
            for j in range(tau*dim*ode_sampling_freq):
                sols.append(yy)
                yy = T(yy)*dt+yy
    
            sol = torch.stack(sols, dim=0)
            Ty = sol[tau*ode_sampling_freq,:,:][:-tau]
            sol = sol[:,:-tau*(dim-1),:]
            sol = sol.T
            DIM = torch.flip(sol[:,:,::tau*ode_sampling_freq],[2])
           # DIM = DIM[:,tau*(dim-1):,:]
            
        
            
    
            if method == 'DIM':        
                L = sample_loss(Ty,Ty_true)+sample_loss(DIM,y_delayed).sum()+torch.mean((DIM[:,0,:-1]-y_delayed[:,0,:-1])**2)
            
            if method == 'IM':
                L = sample_loss(Ty,Ty_true)
                
            if method == 'P':
                L = torch.mean((DIM - y_delayed)**2)
               # L = torch.mean((Ty - Ty_true)**2)
    
    
            L.backward()
            optimizer.step()
            
        
            print('Iteration ',ii)
            print(L)
            if ii%100 == 0:
                
                sols = []
                yy = y.clone()
                for j in range(tau*dim*ode_sampling_freq):
                    sols.append(yy)
                    yy = T(yy)*dt+yy
    
                sol = torch.stack(sols, dim=0)
                Ty = sol[tau*ode_sampling_freq,:,:][:-tau]
                sol = sol[:,:-tau*(dim-1),:]
                sol = sol.T
                DIM = torch.flip(sol[:,:,::tau*ode_sampling_freq],[2])
                
              
            
            
                # plt.scatter(Ty.detach().numpy()[:,0],Ty.detach().numpy()[:,1])
                # plt.scatter(Ty_true.detach().numpy()[:,0],Ty_true.detach().numpy()[:,1])
    
                # plt.show()
                
                fig,ax = plt.subplots(1,6,figsize = (15,3),dpi = 300)
                for i in range(5):
                    ax[i].scatter(DIM.detach().numpy()[i,:,0],DIM.detach().numpy()[i,:,1],s = 2)
                    ax[i].scatter(y_delayed.detach().numpy()[i,:,0],y_delayed.detach().numpy()[i,:,1],s = 2)
                ax[-1].scatter(Ty.detach().numpy()[:,0],Ty.detach().numpy()[:,1],s = 2)
                ax[-1].scatter(Ty_true.detach().numpy()[:,0],Ty_true.detach().numpy()[:,1],s = 2)
    
                plt.show()
                
                
               
                #ground truth
                with torch.no_grad():
                    if ii % 1000 == 0:
                        y_sim = y[0].clone()
                        n_sim = 10000
                        z = torch.zeros((n_sim,len(y[0])))
                        for l in range(n_sim):
                            z[l] = y_sim
                        
                            y_sim = y_sim+dt*T(y_sim)
                        plt.plot(z[:,0].detach().numpy())
                        plt.show()
                
                
                            
    
            #if i %100==0:
            
                
             
    #ground truth
    y_sim = torch.tensor( scaler.transform(testing[0].reshape(1,len(testing[0]))),dtype = torch.float)
    n_sim = int((len(testing)*ode_sampling_freq)//data_sample_rate)
    z = torch.zeros((n_sim,len(y[0])))
    for i in range(n_sim):
        z[i] = y_sim
    
        y_sim = y_sim+dt*T(y_sim)
    
    
    pred = scaler.inverse_transform(z.detach().numpy())
    n_plot_sim =2000
    n_plot_data = data_sample_rate*n_plot_sim//ode_sampling_freq
    dts_sub = [i*dt for i in range(n_sim)]
    dts_data = [i*dt_data for i in range(len(testing))]
    # plt.title('Pointwise')
    plt.plot(dts_sub[:n_plot_sim],pred[:,0][:n_plot_sim])
    plt.plot(dts_data[:n_plot_data],testing[:,0][:n_plot_data])
    plt.show()
    
        
    if method == 'IM':
        with open(f"flow_results_invariant{iii}.p", "wb") as f:
             pickle.dump([pred,ode_sampling_freq,dts_sub,dts_data], f)
    
    if method == 'DIM':
             with open(f"flow_results_delay{iii}.p", "wb") as f:
                  pickle.dump([pred,ode_sampling_freq,dts_sub,dts_data], f)
        
    if method == 'P':
             with open(f"flow_results_pointwise{iii}.p", "wb") as f:
                  pickle.dump([pred,ode_sampling_freq,dts_sub,dts_data], f)
     
    
    
     
    
    
