import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator
from matplotlib.ticker import FormatStrFormatter
import matplotlib.ticker as ticker



##################


########################## read in the clean dataset to generate a POD basis
with open("flow_full70.p", "rb") as f:
    data = pickle.load(f)
  
with open("flow_results_delay0.p", "rb") as f:
    data2 = pickle.load(f)
    
with open("flow_results_invariant0.p", "rb") as f:
    data3 = pickle.load(f)
    
with open("flow_sensors.p", "rb") as f:
     data4 = pickle.load(f)
   

dt_data_sub, dx_phys = data4[-2], data4[-1]     
obs_error = data4[-3]
ode_sampling_freq = data2[1]
training = data4[0]
yM = data2[0]
yP = data3[0]
yT = data4[1]
yN = data4[2]

data_sample_rate = data4[3]
dts,dts_sub = data2[-1], data2[-2]
sampling_freq = data2[-3]

CYLINDER_CENTER_INDEX_X, CYLINDER_CENTER_INDEX_Y,CYLINDER_RADIUS_INDICES = data[1], data[2], data[3]
x = np.arange(150)*dx_phys
y = np.arange(50)*dx_phys
X, Y = np.meshgrid(x, y, indexing="ij")

V = data[0][3000:]

# ixs = [[45,25],[42,20], [42,30], [37,15], [37,35]]

ixs = [[60,16], [52,40], [77,15], [45,35],[75,30],[110,40], [120,15], [80,45],[100,20]]
m = len(ixs)
nt = len(V)


from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

fig, ax = plt.subplots(dpi=300, figsize=(8, 2))

im = ax.contourf(x, y, 100*V[1000].T, alpha=1, cmap='RdBu_r', vmin=0, vmax=0.07*100, levels=300)

# Draw cylinder
cylinder_center = (CYLINDER_CENTER_INDEX_X*dx_phys, CYLINDER_CENTER_INDEX_Y*dx_phys)
ax.add_patch(plt.Circle(cylinder_center, CYLINDER_RADIUS_INDICES*dx_phys, facecolor="white"))

# Add sensors
for i in range(m):
    ax.add_patch(plt.Circle((ixs[i][0]*dx_phys, ixs[i][1]*dx_phys), obs_error*3, color="grey", alpha=0.5))
    ax.scatter(ixs[i][0]*dx_phys, ixs[i][1]*dx_phys, marker='.', s=3, color='k')

# Colorbar
cbar = fig.colorbar(im, ax=ax, label='Flow Speed', shrink=1, pad=0.01)
cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
cbar.set_ticks([0,1,2,3,4,5,6])

ax.set_title('Flow Past Cylinder', fontsize=12)
ax.set_xlabel(r'$x$', fontsize=12,labelpad = -1)
ax.set_ylabel(r'$y$', fontsize=12,rotation = 0,labelpad = 8)

# -------- Add Time Series as Inset --------
inset_position = [0.6, 1.03, 0.25, 0.15]  # adjust manually here
inset_ax = fig.add_axes(inset_position)

inset_ax.plot([i*dt_data_sub for i in range(len(training))], training[:, -1], color='k', linewidth=1)
inset_ax.set_xlabel(r'$t$', fontsize=8,labelpad = -0.5)
inset_ax.set_ylabel(r'$S_7(t)$', fontsize=8,rotation = 0,labelpad = 10)
inset_ax.yaxis.set_label_coords(-0.15, 0.3)
inset_ax.tick_params(axis='both', labelsize=6)

# -------- Add Connection Line --------
# Add annotation (line) from cylinder center to inset
ax.annotate("",
            xy=(0, 0),  # relative coords of inset location
            xytext = (10,2),
            xycoords=inset_ax.transAxes,
            textcoords='data',
            arrowprops=dict(arrowstyle="->", color="black", lw=1))

plt.show()


n_plot_sim =1000
n_plot_data = data_sample_rate*n_plot_sim//ode_sampling_freq


fig,ax = plt.subplots(4,1,figsize = (8,2.5),dpi = 300)
for i in range(4):
    ax[i].plot(dts_sub[:n_plot_sim],yM[:,i][:n_plot_sim],'--',color = 'b',linewidth = 1.5, label = 'Delay Measure Forecast' ,alpha = 0.5)
    ax[i].plot(dts_sub[:n_plot_sim],yP[:,i][:n_plot_sim],'--',color = 'r',linewidth = 1.5, label = 'State Measure Forecast',alpha = 0.5)
    ax[i].plot(dts[:n_plot_data],yT[:,i][:n_plot_data],color = 'grey',linewidth = 3,alpha = 0.5,label = 'Ground Truth',zorder = -100)

    if i < 3:
        ax[i].set_xticks([])
plt.subplots_adjust(wspace = 0.25,hspace = 0)
ax[0].legend(
    loc='lower left',
    bbox_to_anchor=(0, 1.02),  # Adjusts the position above the plot (x, y)
    ncol=3,                    # Makes it horizontal (3 items side by side)
    fontsize=10,
    frameon=False
)
for i in range(1):
    ax[0].set_ylim(2.5,6.8)
    ax[1].set_ylim(4.1,5.4)
    ax[2].set_ylim(2.01,6.2)
    ax[3].set_ylim(4.01,6.5)




ax[-1].set_xlabel(r'$t$',fontsize = 12)
ax[0].set_ylabel(r'$s_1(t)$',fontsize = 11,rotation = 0,labelpad = 10)
ax[1].set_ylabel(r'$s_2(t)$',fontsize = 11,rotation = 0,labelpad = 10)
ax[2].set_ylabel(r'$s_3(t)$',fontsize = 11,rotation = 0,labelpad = 10)
ax[3].set_ylabel(r'$s_4(t)$',fontsize = 11,rotation = 0,labelpad = 10)

for i in range(4):
        ax[i].tick_params(axis='both', labelsize=8) 
        ax[i].yaxis.set_label_coords(-.08, .3)
       # ax[i].yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}"))


plt.show()

