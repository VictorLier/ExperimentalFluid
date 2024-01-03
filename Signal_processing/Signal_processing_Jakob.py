from scipy.io import loadmat
mat = loadmat('gridturbulence.mat')
u = mat['u']
fs = mat['fs'][0,0] #convert to value
n, nblock = u.shape
dt = 1 / fs

