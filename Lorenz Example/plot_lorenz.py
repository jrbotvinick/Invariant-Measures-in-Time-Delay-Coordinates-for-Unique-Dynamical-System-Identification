import numpy as np
import matplotlib.pyplot as plt
import pickle

with open("IM_Recon2.p", "rb") as f:
    data = pickle.load(f)
 
with open("DIM_Recon2.p", "rb") as f:
    data2 = pickle.load(f)

y,Ty,y_delay,samples,xs_IM,Ty1,DIM1,losses = data[0],data[1],data[2],data[3], data[4], data[5],data[6], data[7]
y,Ty,y_delay,samples,xs_DIM,Ty2,DIM2,losses = data2[0],data2[1],data2[2],data2[3], data2[4], data2[5],data2[6], data2[7]


length = int(1e6)
fig,ax = plt.subplots(3,3,figsize = (6,3.5),dpi = 300)
ax[0,0].plot(y[:,0][:length],y[:,2][:length],color = 'k',linewidth = .0075)
ax[0,1].plot(xs_IM[:,0][:length],xs_IM[:,2][:length],linewidth = .5,color = 'k')
#ax[0,2].plot(y_I[:,0][:length],y_I[:,2][:length],linewidth = 0.005,color = 'k')

ax[0,2].plot(xs_DIM[:,0][:length],xs_DIM[:,2][:length],linewidth = 0.0075,color = 'k')


for i in range(3):
    ax[0,i].set_xlim(-25,25)
    ax[0,i].set_ylim(2,52)
    ax[1,i].set_xlim(-25,25)
    ax[1,i].set_ylim(2,52)


    #ax[0,i].set_xticks([])
    ax[1,i].set_xticks([])

   
ax[0,1].set_yticks([])
ax[0,2].set_yticks([])
ax[2,1].set_yticks([])
ax[2,2].set_yticks([])

#ax[0,3].set_yticks([])

ax[1,1].set_yticks([])
ax[1,2].set_yticks([])
#ax[1,3].set_yticks([])

plt.subplots_adjust(wspace = .04,hspace = .08)

cm = 'viridis'
vmax = 0.005
ranges = [[-25,25],[2,52]]
ranges2 = [[-25,25],[-25,25]]

bins = 75
ax[1,0].hist2d(Ty[:,0],Ty[:,2],bins = bins,range = ranges,vmin = 0,vmax = vmax,cmap = cm,density = True)
ax[1,1].hist2d(Ty1[:,0],Ty1[:,2],bins = bins,range = ranges,vmin = 0,vmax = vmax,cmap = cm,density = True)
#ax[1,2].hist2d(y_I[:,0],y_I[:,2],bins = 50,range = ranges,vmin = 0,vmax = vmax,cmap = cm,density = True)
im = ax[1,2].hist2d(Ty2[:,0],Ty2[:,2],bins = bins,cmap = cm,range = ranges,vmin = 0,vmax = vmax,density = True)
ax[2,0].hist2d(y_delay[:,0],y_delay[:,1],bins = bins,range = ranges2,vmin = 0,vmax = vmax,cmap = cm,density = True)
ax[2,1].hist2d(DIM1[:,0],DIM1[:,1],bins = bins,range = ranges2,vmin = 0,vmax = vmax,cmap = cm,density = True)
ax[2,2].hist2d(DIM2[:,0],DIM2[:,1],bins = bins,range = ranges2,vmin = 0,vmax = vmax,cmap = cm,density = True)

ax[0,0].set_title('Ground Truth',fontsize = 9)
ax[0,1].set_title(r'Invariant Measure Rec.',fontsize = 9)
#ax[0,2].set_title('Invariant Measure Rec.',fontsize = 10)
ax[0,2].set_title(r'Delay Measure Rec.',fontsize = 9)


cbar_ax = fig.add_axes([0.92, 0.12, 0.01, 0.75])  # <-- adjust these values

cbar = fig.colorbar(im[3], cax=cbar_ax)
cbar.set_label("Mass Per Cell", fontsize=9)
cbar.ax.tick_params(labelsize=7)
for i in range(3):
    for j in range(3):
        
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
#ax[1,0].set_title('Ground Truth Statistics',fontsize = 10)
#ax[1,1].set_title('Pointwise-Learned Statistics',fontsize = 10)
#ax[1,2].set_title('Invariant Measure-Learned Statistics',fontsize = 10)
#ax[1,3].set_title('Delay Measure-Learned Statistics',fontsize = 10)
ax[0,0].set_ylabel('Trajectory',fontsize = 10)
ax[1,0].set_ylabel(r'$T\#\mu$',fontsize = 11)
ax[2,0].set_ylabel(r'$\Psi \# \mu$',fontsize = 11)

plt.show()