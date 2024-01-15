# -*- coding: utf-8 -*-
"""
Process LDA signal into detected velocities

version 0.1,  15. Januar 2024

@author: keme@dtu.dk
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, lfilter, hilbert
from time import time

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def gauss_interpolate(x):
    """ Return fractional peak position assuming Guassian shape
        x is a numpy vector with 3 elements,
        ifrac returned is relative to center element.
    """
    from numpy import log
    assert (x[1] >= x[0]) and (x[1] >= x[2]), 'Peak must be at center element'
    # avoid log(0) or divide 0 error
    if all(x > 0):
        r = log(x)
        ifrac = (r[0] - r[2]) / (2 * r[0] - 4 * r[1] + 2 * r[2])
    else:
        # print("using centroid in gauss_interpolate")
        ifrac = (x[2] - x[0]) / sum(x)
    return ifrac

time1 = time()
# measurement parameters
sample_rate = 7.8e6
optical_shift = 0.2e6
fdcal = 0.1877e6      #doppler velocity to frequency calibration
filename = 'data/20231213-100_015_0001_02'

#processing parameters
gain = 20.0
filter_center = 1.45e6
filter_width = 0.5e6
min_periods = 15
max_periods = 100
win_width_Hilbert = 200
burst_trigger = 1.0
burst_schmitt = 0.1
gaussfit_width = 4

#get signal data from data file
data1 = scipy.io.loadmat(filename + '.mat')
sig = data1['A'][0,:] 
np.nan_to_num(sig, copy=False, neginf=-1) # remove -inf in data
sig = gain * sig
dt = 1 / sample_rate
times = np.arange(len(sig)) * dt

# filter signal and find envelope
band = [filter_center - 0.5 * filter_width, filter_center + 0.5 * filter_width]
sigfilt = butter_bandpass_filter(sig, band[0], band[1], 7.8e6)
sig_hilbert = np.abs(hilbert(sigfilt))
sig_envelope = np.convolve(sig_hilbert, 
                           np.ones(win_width_Hilbert)/win_width_Hilbert,
                           mode='same')

# plot the filteret signal and envelope
plt.figure(1)
plt.clf()
plt.plot(times, sigfilt)
plt.plot(times, sig_envelope)

# apply Schmidt trigger to detect bursts
istart = []
iend = []
schmitt_not_high = sig_envelope < burst_trigger + burst_schmitt
schmitt_low = sig_envelope < burst_trigger - burst_schmitt
i = win_width_Hilbert // 2 # start when envelope is established
istop = len(sig) - win_width_Hilbert // 2
while i < istop:
    while i < istop and schmitt_not_high[i]: i += 1
    if i == istop: break
    istart.append(i)
    while i < istop and not schmitt_low[i]: i += 1
    if i == istop: break
    iend.append(i)  
if len(istart) > len(iend): istart.pop()  #remove possible incomplete burst

# plot detected regions with a line
for ii in zip(istart, iend):
    plt.plot(np.array(ii) / sample_rate, [burst_trigger, burst_trigger], 'k-')

# find frequency in bursts and validate
frequency = []
arrival_time = []
transit_time = []
validated = []
nw = 512
f = np.arange(nw) / (nw * dt)
#ipick = int(np.round(0.386837436/ dt)) # select a burst spectrum to plot
ipick = istart[9]
for i, j in zip(istart, iend):
    if j-i > nw: break  # burst too long to handle
    window = np.zeros(nw)
    window[0:j-i] = sigfilt[i:j]
    sigf = np.fft.fft(window)
    sigspec = np.real(sigf * sigf.conjugate()) / (nw*dt)
    ipeak = np.argmax(sigspec)
    ifrac = gauss_interpolate(sigspec[ipeak-1:ipeak+2])
    freq = (ipeak + ifrac) / (nw * dt)
    transit = (j - i) * dt
    nfringes = freq * transit
    frequency.append(freq)
    arrival_time.append(i * dt)
    transit_time.append(transit)
    validated.append(min_periods <= nfringes and nfringes < max_periods)
    if i == ipick:
        plt.figure(2)
        plt.clf()
        plt.plot(f[1:nw//2], sigspec[1:nw//2], '-o')
        
        velocity = (freq - optical_shift) / fdcal
        plt.title("t = %9.7f s, velocity = %6.3f m/s" %
                   (i * dt, velocity))

# write validated velocities to file
with open(filename + '.txt', 'w') as f:
    f.write('# arrival time [s]   velocity [m/s]   transit time [s]\n')
    validationrate = 100 * np.count_nonzero(validated) / len(validated)
    f.write('# validation rate: {:.1f}%\n'.format(validationrate))
    for i in range(len(frequency)):
        if validated[i]:
            velocity = (frequency[i] - optical_shift ) / fdcal
            f.write('{:.9f}  {:.3f}  {:.7f}\n'.format( 
                    arrival_time[i], velocity, transit_time[i]))
        
print('Time used: %4.2f s' % (time()-time1))        