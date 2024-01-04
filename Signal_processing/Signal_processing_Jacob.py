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

uf = u - u.mean() # fluctuation part
autocorr = np.correlate(uf[:,0], uf[:,0], mode='full') / n
plt.figure()
plt.plot(np.arange(n)*dt, autocorr[n-1:])
plt.xlabel('Tau [s]')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of velocity fluctuations for first block')

autocorr_list = np.array([np.correlate(uf[:,b], uf[:,b], mode='full') / n for b in range(nblock)])
autocorr_mean = np.mean(autocorr_list, axis=0)

plt.figure()
plt.plot(np.arange(n)*dt, autocorr_mean[n-1:])
plt.xlabel('Tau [s]')
plt.ylabel('Autocorrelation')
plt.title('Autocorrelation of velocity fluctuations for avarage of all blocks')

# Normilize the autocorrelation
autocorr_norm = autocorr_mean / autocorr_mean.max()
plt.figure()
plt.plot(np.arange(n)*dt, autocorr_norm[n-1:])
plt.xlabel('Tau [s]')
plt.ylabel('Autocorrelation')
plt.title('Normalized autocorrelation of velocity fluctuations for avarage of all blocks')

# Integrate the autocorrelation
integrated_autocorr = np.trapz(autocorr_norm[n-1:], dx=dt)
integrated_autocorr_test = np.cumsum(autocorr_norm[n-1:]*dt)[-1]

# Finding the variability og the mean velocity for each block
u_var_mean = u.var(axis=0)
epsilon_U_block = np.sqrt(1/n * u_var_mean/u_mean_block**2)
u_ind = u[::int(integrated_autocorr/dt),:]
u_ind_mean = u_ind.mean(axis=0)
u_ind_var = u_ind.var(axis=0)
epsilon_U_ind = np.sqrt(1/n * u_ind_var/u_ind_mean**2)

# Compare the two
plt.figure()
plt.plot(np.arange(nblock), epsilon_U_block, label='Block')
plt.plot(np.arange(nblock), epsilon_U_ind, label='Independent')
plt.xlabel('Block number')
plt.ylabel('Variability of the mean velocity')
plt.legend()
plt.title('Variability of the mean velocity for each block')


# Spectrum using FFT on a single block
f = np.arange(n) / (n * dt)
uft = dt * np.fft.fft(u[:,0])
S = uft * uft.conjugate() / (n * dt)
plt.figure()
plt.loglog(f[0:n//2], S[0:n//2].real, 'k-')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power spectrum [m^2/s]')
plt.title('Power spectrum for first block')

# Spectrum using FFT on all blocks
uft_list = dt * np.fft.fft(u, axis=0)
S_list = uft_list * uft_list.conjugate() / (n * dt)
S_mean = np.mean(S_list, axis=1)
plt.figure()
plt.loglog(f[0:n//2], S_mean[0:n//2].real, 'k-')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power spectrum [m^2/s]')
plt.title('Power spectrum for all blocks')

S0 = 2*np.mean(u_var_mean)*integrated_autocorr

n_sample = [200, 400, 600, 800, 1000, 1200]
plt.figure()
for ns in n_sample:
    f_sample = np.arange(ns) / (ns * dt)
    u_sample = u[:ns, :]
    uft_sample = dt * np.fft.fft(u_sample, axis=0)
    S_sample = uft_sample * uft_sample.conjugate() / (ns * dt)
    S_sample_mean = np.mean(S_sample, axis=1)

    plt.loglog(f_sample[1:ns//2], S_sample_mean[1:ns//2].real, label=f'Sample size: {ns}')

plt.xlabel('Frequency [Hz]')
plt.ylabel('Power spectrum [m^2/s]')
plt.title('Power spectrum for different sample sizes')
plt.legend()

n_step = [1, 2, 3, 4, 5, 6]
plt.figure()
for step in n_step:
    u_sample = u[::step, :]
    ns, _ = u_sample.shape
    f_sample = np.arange(ns) / (ns * dt * step)
    uft_sample = dt * step * np.fft.fft(u_sample, axis=0)
    S_sample = uft_sample * uft_sample.conjugate() / (ns * dt * step)
    S_sample_mean = np.mean(S_sample, axis=1)

    plt.loglog(f_sample[0:ns//2], S_sample_mean[0:ns//2].real, label=f'Step size: {step}')

plt.xlabel('Frequency [Hz]')
plt.ylabel('Power spectrum [m^2/s]')
plt.title('Power spectrum for different step sizes')
plt.legend()

n_blocks = [100, 200, 400, 600, 800, 1000]
plt.figure()
for blocks in n_blocks:
    u_sample = u[:, :blocks]
    ns, _ = u_sample.shape
    f_sample = np.arange(ns) / (ns * dt)
    uft_sample = dt * np.fft.fft(u_sample, axis=0)
    S_sample = uft_sample * uft_sample.conjugate() / (ns * dt)
    S_sample_mean = np.mean(S_sample, axis=1)

    plt.loglog(f_sample[0:ns//2], S_sample_mean[0:ns//2].real, label=f'Blocks: {blocks}')

plt.xlabel('Frequency [Hz]')
plt.ylabel('Power spectrum [m^2/s]')
plt.title('Power spectrum for different numbers of blocks')
plt.legend()
plt.show()

plt.show()
