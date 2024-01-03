from scipy.io import loadmat
mat = loadmat('Signal_processing\gridturbulence.mat')
u = mat['u']
fs = mat['fs'][0,0] #convert to value
n, nblock = u.shape
dt = 1 / fs


import matplotlib.pyplot as plt
plt.plot(u[:,0])
plt.show()