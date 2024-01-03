from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

mat = loadmat(r'Signal_processing\gridturbulence.mat')
u = mat['u']
fs = mat['fs'][0,0] #convert to value
n, nblock = u.shape
dt = 1 / fs

uf = u - u.mean() # fluctuation part
autocorr = np.correlate(uf[:,0], uf[:,0], mode='full') / n
plt.figure()
plt.plot(np.arange(n)*dt, autocorr[n-1:])
plt.show()