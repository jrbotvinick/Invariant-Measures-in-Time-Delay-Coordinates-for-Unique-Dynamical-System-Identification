import numpy as np
import pickle
import matplotlib.pyplot as plt
from ot.sliced import sliced_wasserstein_distance
import numpy as np

DATA_IM = []
DATA_DIM = []
for i in range(10):
    with open(f"IM_Recon{i}.p", "rb") as f:
        data = pickle.load(f)
    y = data[0]
    DATA_IM.append(data[4])
    with open(f"DIM_Recon{i}.p", "rb") as f:
        data = pickle.load(f)
    DATA_DIM.append(data[4])
    
    
    
plt.scatter(y[:,0],y[:,1],s = 1)
plt.show()


errs_IM = []
errs_DIM = []
every = 100

for i in range(10):
    dim = sliced_wasserstein_distance(y[::every],DATA_DIM[i][::every])
    errs_DIM.append(dim)
    
    im = sliced_wasserstein_distance(y[::every],DATA_IM[i][::every])
    errs_IM.append(im)

print('Delay Measure mean:', np.mean(errs_DIM), 'Measure STD:', np.std(errs_DIM))
print('Invariant measure mean:', np.mean(errs_IM), 'Pointwise STD:', np.std(errs_IM))


        