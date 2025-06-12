import pickle
import numpy as np
import matplotlib.pyplot as plt 
import numpy as np
from ot.sliced import sliced_wasserstein_distance

with open("K.p", "rb") as f:
    data_load = pickle.load(f)


data = data_load[0]
gt_full = data_load[1]
dt = 0.1
ts = [i*dt for i in range(int(3e3))]
xs = np.linspace(0,99.5,200)
plt.figure(dpi = 300)
plt.title('Full Dynamics',fontsize = 15)
plt.contourf(ts,xs,gt_full[:int(3e3)].T,levels = 300,origin = 'lower',cmap = 'RdBu_r')
cbar = plt.colorbar()
cbar.set_label(r'$u(t,x)$',fontsize = 15)
plt.xlabel(r'$t$',fontsize = 15)
plt.ylabel(r'$x$',fontsize = 15)
plt.show()

print(len(data))
def delay(X,dim,tau):
    new = np.zeros((len(X)-tau*dim,dim))
    for i in range(dim):
        new[:,i] = X[dim*tau-(i+1)*tau:-(1+i)*tau]
    return new

def MLoss(s1,s2):
    dim,tau = 5, 1
    ds1 = delay(s1,dim,tau)
    ds2 = delay(s2,dim,tau)
    return sliced_wasserstein_distance(ds1,ds2)
    
      
    
def L2(s1,s2):
    return np.mean((s1-s2)**2)    



ix = 0
freq = 30





scale = 0.25
gt_clean = gt_full[:,0]
gt = gt_full[::freq,0]
noise = np.random.normal(0,scale,len(gt))
gt = gt+noise

N_plot = freq*300
dts = [dt*i for i in range(N_plot)]
dts2 = [dt*freq*i for i in range(N_plot//freq)]
plt.figure(dpi = 300)
plt.plot(dts,gt_clean[:N_plot],linewidth = 0.75,color = 'k',label = 'Full time-series')
plt.scatter(dts2,gt[:N_plot//freq],s = 4,color = 'salmon',label = 'Noisy observations')
plt.xlabel(r'$t$',fontsize = 15)
plt.ylabel(r'$y(t)$',fontsize = 15)
plt.title('Partial Observation',fontsize = 15)
plt.legend(loc = 'upper left')
plt.show()

pointwise = []
measure = []

for i in range(len(data)-1):
    print(i)
    st = data[i][::freq]
    pointwise.append(L2(st,gt))
    measure.append(MLoss(st,gt))
    
  

  
params = np.linspace(0.5,1.5,100).tolist()

plt.figure(dpi = 300)
plt.title('Optimization Landscape',fontsize = 15)
plt.plot(params,pointwise,'-o',markersize = 3,linewidth = .5,color = 'darkorange',alpha = .75,label = 'Pointwise Matching')
plt.ylabel('Obejctive Function',fontsize = 15)
plt.plot(params,measure,'-s',markersize = 3,linewidth = .5,color = 'steelblue',alpha = .75,label = 'Measure Matching')
plt.axvline(x = 1,linestyle = '--',linewidth = 2,color = 'grey',label = 'True parameter',alpha = .75)
plt.xlabel(r'Parameter',fontsize= 15)
plt.yscale('log')
plt.legend(loc = 'lower left')
plt.show()


with open("landscape.p", "wb") as f:
     pickle.dump([params,pointwise,measure], f)
