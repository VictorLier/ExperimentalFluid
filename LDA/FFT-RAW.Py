import numpy as np
import matplotlib.pyplot as plt
import scipy
import os

parent_dir = os.path.dirname(os.getcwd())
data_path = os.path.join(parent_dir, r'LDA_DATA\20240117-0001_test3_slowjet\20240117-0001_test3_slowJet_02')

data1 = scipy.io.loadmat(data_path + '.mat')

sig = data1['A'][0,:]
sig = sig[len(sig)//10:]
np.nan_to_num(sig, copy=False, neginf=-1) # remove -inf in data

sample_rate = 7.8e6
nw = len(sig)

dt = 1 / sample_rate


f = np.arange(nw) / (nw * dt)



times = np.arange(len(sig)) * dt



sigf = np.fft.fft(sig)

sigspec = np.real(sigf * sigf.conjugate()) / (nw*dt)

plt.figure(1)
plt.loglog(f[1:nw//2], sigspec[1:nw//2])
plt.show()


