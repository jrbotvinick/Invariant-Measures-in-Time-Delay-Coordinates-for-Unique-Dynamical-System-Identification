import numpy as np
import pickle
import matplotlib.pyplot as plt
from ot.sliced import sliced_wasserstein_distance
import numpy as np

with open("flow_sensors.p", "rb") as f:
     data4 = pickle.load(f)

yT = data4[1]

DATA_P = []
DATA_DIM = []
for i in range(10):
    with open(f"flow_results_delay{i}.p", "rb") as f:
        data = pickle.load(f)
    pred,ode_sampling_freq,dts_sub,dts_data = data[0], data[1], data[2], data[3]
    plt.plot(pred[:1000,0])
    plt.plot(yT[:1000,0])
    plt.show()
  #  plt.plot(pred[:1000,0])
    DATA_DIM.append(pred)
    with open(f"flow_results_invariant{i}.p", "rb") as f:
        data = pickle.load(f)
   
    pred,ode_sampling_freq,dts_sub,dts_data = data[0], data[1], data[2], data[3]
    plt.plot(pred[:1000,0])
    plt.plot(yT[:1000,0])
    plt.show()
    DATA_P.append(pred)
    
    
   
    

dt_sim = dts_sub[1] - dts_sub[0]
dt_true = dts_data[1] - dts_data[0]
every = int(dt_true//dt_sim)

errs_DIM = []
errs_P = []

for i in range(10):
    ed = np.mean(np.linalg.norm(DATA_DIM[i][::every][:1000] - yT[:1000],axis = 1))
    ep = np.mean(np.linalg.norm(DATA_P[i][::every][:1000] - yT[:1000],axis = 1))
    errs_DIM.append(ed)
    errs_P.append(ep)
    
print('Delay Measure mean:', np.mean(errs_DIM), 'Measure STD:', np.std(errs_DIM))
print('Invariant mean:', np.mean(errs_P), 'Invariant STD:', np.std(errs_P))

