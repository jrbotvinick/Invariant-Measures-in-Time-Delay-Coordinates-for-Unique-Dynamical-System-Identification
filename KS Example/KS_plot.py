import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable

with open("K.p", "rb") as f:
    data_load = pickle.load(f)

with open("landscape.p", "rb") as f:
    data = pickle.load(f)

with open("KS_opt_results.p", "rb") as f:
    data2 = pickle.load(f)


import numpy as np
import matplotlib.pyplot as plt 
import numpy as np
from geomloss import SamplesLoss
import torch
from ot.sliced import sliced_wasserstein_distance

losses,losses2,res,res2 = data2[0], data2[1], data2[2], data2[3]
losses = np.array([i[0] if isinstance(i, np.ndarray) else i for i in losses])
losses2 = np.array([i[0] if isinstance(i, np.ndarray) else i for i in losses2])

params, pointwise, measure = data[0], data[1], data[2]

data = data_load[0]
gt_full = data_load[1]
dt = 0.1
ts = [i*dt for i in range(int(3e3))]
xs = np.linspace(0,99.5,200)

fig,ax = plt.subplots(2,2,figsize = (12,8),dpi = 300)

ax[0,0].set_title('Full Dynamics',fontsize = 15)
im = ax[0,0].contourf(ts,xs,gt_full[:int(3e3)].T,levels = 300,origin = 'lower',cmap = 'RdBu_r')
divider = make_axes_locatable(ax[0,0])

cax = divider.append_axes("right", size="3%", pad=0.1)  # 5% width, 0.05 pad
cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r'$u(t,x)$', fontsize=15,labelpad = -.5)
ticks = [-3, -2,-1,0, 1,2,3]  # customize as needed based on your data range
cbar.set_ticks(ticks)
ax[0,0].set_xlabel(r'$t$',fontsize = 15)
ax[0,0].set_ylabel(r'$x$',fontsize = 15)

freq = 30
scale = 0.25
gt_clean = gt_full[:,0]
gt = gt_full[::freq,0]
noise = np.random.normal(0,scale,len(gt))
gt = gt+noise

N_plot = freq*300
dt = 0.1
dts = [dt*i for i in range(N_plot)]
dts2 = [dt*freq*i for i in range(N_plot//freq)]
ax[0,1].plot(dts,gt_clean[:N_plot],linewidth = 0.75,color = 'k',label = 'Full time-series')
ax[0,1].scatter(dts2,gt[:N_plot//freq],s = 4,color = 'salmon',label = 'Noisy observations')
ax[0,1].set_xlabel(r'$t$',fontsize = 15)
ax[0,1].set_ylabel(r'$y(t)$',fontsize = 15,labelpad = -.5)
ax[0,1].set_title('Partial Observation',fontsize = 15)
ax[0,1].legend(loc = 'upper right')

ax[1,0].set_title('Optimization Landscape',fontsize = 15)
ax[1,0].plot(params,pointwise,'-o',markersize = 3,linewidth = .5,color = 'darkorange',alpha = .75,label = 'Pointwise')
ax[1,0].set_ylabel('Obejctive Function',fontsize = 15)
ax[1,0].plot(params,measure,'-s',markersize = 3,linewidth = .5,color = 'steelblue',alpha = .75,label = 'Measure')
ax[1,0].axvline(x = 1,linestyle = '--',linewidth = 2,color = 'grey',label = 'Truth',alpha = .75)
ax[1,0].set_xlabel(r'Parameter',fontsize= 15)
ax[1,0].set_yscale('log')
ax[1,0].legend(loc = 'upper right')


plt.subplots_adjust(wspace = 0.3,hspace = .3)

ax[1,1].set_title('System Identification',fontsize = 15)
ax[1,1].plot(losses2,'-o',markersize = 5,linewidth = 1,color = 'darkorange',alpha = .75,label = 'Pointwise')
ax[1,1].plot(losses,'-s',markersize = 5,linewidth = 1,color = 'steelblue',alpha = .75,label = 'Measure')

ax[1,1].set_xlabel('Iterations',fontsize = 15)
ax[1,1].set_ylabel('Parameter',fontsize = 15)
ax[1,1].axhline(y = 1,linestyle = '--',linewidth = 2,color = 'grey',label = 'Truth',alpha = .75)
ax[1,1].legend(loc = 'center right')
ax[1,1].set_ylim(0.4,1.1)
plt.show()






