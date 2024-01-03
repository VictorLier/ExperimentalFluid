from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

# Part 1 (slide 1-13)
mat = loadmat(r'Signal_processing\gridturbulence.mat')
u = mat['u']
fs = mat['fs'][0,0] #convert to value
n, nblock = u.shape
dt = 1 / fs

block_plot = 0
plt.figure()
plt.plot(np.arange(n)*dt + block_plot*dt*n, u[:,block_plot])
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.title('Velocity time series')

u_mean_block = u.mean(axis=0)
plt.figure()
plt.plot(np.arange(nblock), u_mean_block)
plt.xlabel('Block number')
plt.ylabel('Mean velocity [m/s]')
plt.title('Mean velocity per block')



# uf = u - u.mean() # fluctuation part
# autocorr = np.correlate(uf[:,0], uf[:,0], mode='full') / n
# plt.figure()
# plt.plot(np.arange(n)*dt, autocorr[n-1:])
plt.show()