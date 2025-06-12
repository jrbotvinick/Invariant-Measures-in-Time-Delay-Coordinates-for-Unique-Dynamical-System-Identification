import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


##################
with open("flow_full70.p", "rb") as f:
    data = pickle.load(f)
    


CYLINDER_CENTER_INDEX_X, CYLINDER_CENTER_INDEX_Y,CYLINDER_RADIUS_INDICES,PLOT_EVERY = data[1], data[2], data[3], data[4]

  
#### PARAMETERS
data_sample_rate = 10
n_train = 200
dx_phys = 0.1
dv_phys = 100
dt_phys = dx_phys/dv_phys
dt_data = dt_phys*PLOT_EVERY
dt_data_sub = dt_phys*PLOT_EVERY*data_sample_rate
obs_error = 0.5*dx_phys

x = np.arange(150)*dx_phys
y = np.arange(50)*dx_phys
X, Y = np.meshgrid(x, y, indexing="ij")
V = data[0][3000:]*dv_phys
ixs = [[62,12], [52,44], [77,15], [43,35],[74,35],[109,40], [125,15], [80,35],[100,20]]
m = len(ixs)
nt = len(V)

fig, ax = plt.subplots(dpi = 300,figsize =(8,2))
im = ax.contourf(x,y,V[1000].T,alpha = 1,cmap = 'RdBu_r',vmin = 0,vmax = dv_phys*0.07,levels = 100)
ax.add_patch(plt.Circle(
             (CYLINDER_CENTER_INDEX_X*dx_phys, CYLINDER_CENTER_INDEX_Y*dx_phys),
             CYLINDER_RADIUS_INDICES*dx_phys,
             facecolor="white"))
cbar = fig.colorbar(im, ax=ax,label = 'Flow Speed', shrink=1, pad=0.01)

observed_true = np.zeros((len(V),m))
for i in range(m):
    observed_true[:,i] = V[:,ixs[i][0],ixs[i][1]]
    
   # ax.contour(X,Y,gaussian)
    ax.add_patch(plt.Circle(
                 (ixs[i][0]*dx_phys,ixs[i][1]*dx_phys),
                 obs_error*3,
                 color="grey",alpha = 0.75 ))
    plt.scatter(ixs[i][0]*dx_phys,ixs[i][1]*dx_phys,marker = '.',s = 3,color = 'k')
    
plt.title('Flow Past Cylinder',fontsize = 12)
plt.xlabel(r'$x$',fontsize = 12)
plt.ylabel(r'$y$',fontsize = 12)    
    
plt.show()

ads = [[0,0],[0,1],[1,0],[-1,-1],[-1,0],[0,-1],[1,1],[1,-1],[-1,1]]

observed_noise = np.zeros((len(V),m))


for i in range(m):
    for j in range(len(V)):
        random = np.random.normal(np.array(ixs[i])*dx_phys,obs_error)
        interp = RegularGridInterpolator((x,y), V[j])  
        observed_noise[j,i] = interp(random)[0]

for i in range(m):
    plt.figure(dpi = 300)
    plt.plot(observed_true[:,i][::data_sample_rate][:200])
    plt.plot(observed_noise[:,i][::data_sample_rate][:200])


    plt.show()



states = observed_noise.copy()
states_testing = observed_true.copy()

plt.scatter(states[:,0],states[:,1],s = 1)
plt.scatter(states_testing[:,0],states_testing[:,1],s = 1)

plt.show()

training = states[::data_sample_rate][:n_train]
testing = states_testing[n_train*data_sample_rate:]
testing_noised = states[n_train*data_sample_rate:]

with open("flow_sensors.p", "wb") as f:
         pickle.dump([training,testing,testing_noised,data_sample_rate,dt_data,obs_error,dt_data_sub,dx_phys], f)
       
