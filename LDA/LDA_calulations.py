import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import butter, lfilter, hilbert
from time import time
from numpy import log
import os

class LDA_calcs():
    def __init__(self, data_path: str, proc_data_path: str) -> None:

        # data paths
        self._data_path = data_path
        self._proc_data_path = proc_data_path
        self._filename = self._get_filename_list()

        # measurement parameters
        self._sample_rate = 7.8e6
        self._optical_shift = 0.2e6
        self._fdcal = 0.1877e6      #doppler velocity to frequency calibration

        #processing parameters
        self._gain = 20.0
        self._filter_center = 1.45e6
        self._filter_width = 0.5e6
        self._min_periods = 15
        self._max_periods = 100
        self._win_width_Hilbert = 200
        self._burst_trigger = 1.0
        self._burst_schmitt = 0.1
        self._gaussfit_width = 4

        # filter signal and find envelope
        self._band = [self._filter_center - 0.5 * self._filter_width, self._filter_center + 0.5 * self._filter_width]

        self._nw = 512

    def _get_filename_list(self):
        """ Return list of filenames in data_path"""
        filenames = []
        for file in os.listdir(self._data_path):
            if file.endswith(".mat"):
                filenames.append(file[:-4])
        return filenames

    def _butter_bandpass(self, lowcut, highcut, fs, order=5):
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype='band')
        return b, a

    def _butter_bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        b, a = self._butter_bandpass(lowcut, highcut, fs, order=order)
        y = lfilter(b, a, data)
        return y

    def _gauss_interpolate(self, x):
        """ Return fractional peak position assuming Guassian shape
            x is a numpy vector with 3 elements,
            ifrac returned is relative to center element.
        """
        assert (x[1] >= x[0]) and (x[1] >= x[2]), 'Peak must be at center element'
        # avoid log(0) or divide 0 error
        if all(x > 0):
            r = log(x)
            ifrac = (r[0] - r[2]) / (2 * r[0] - 4 * r[1] + 2 * r[2])
        else:
            # print("using centroid in gauss_interpolate")
            ifrac = (x[2] - x[0]) / sum(x)
        return ifrac
    
    def eval_file(self, filename: str, plot: bool = False):
        """ Evaluate data file"""
        #get signal data from data file
        data_f = os.path.join(self._data_path, filename)
        data1 = scipy.io.loadmat(data_f + '.mat')
        sig = data1['A'][0,:] 
        np.nan_to_num(sig, copy=False, neginf=-1) # remove -inf in data
        sig = self._gain * sig
        dt = 1 / self._sample_rate
        times = np.arange(len(sig)) * dt

        sigfilt = self._butter_bandpass_filter(sig, self._band[0], self._band[1], 7.8e6)
        sig_hilbert = np.abs(hilbert(sigfilt))
        sig_envelope = np.convolve(sig_hilbert, 
                                    np.ones(self._win_width_Hilbert)/self._win_width_Hilbert,
                                    mode='same')
        
        if plot:
            plt.figure(1)
            plt.clf()
            plt.plot(times, sigfilt)
            plt.plot(times, sig_envelope)

        # apply Schmidt trigger to detect bursts
        istart = []
        iend = []
        schmitt_not_high = sig_envelope < self._burst_trigger + self._burst_schmitt
        schmitt_low = sig_envelope < self._burst_trigger - self._burst_schmitt
        i = self._win_width_Hilbert // 2 # start when envelope is established
        istop = len(sig) - self._win_width_Hilbert // 2
        while i < istop:
            while i < istop and schmitt_not_high[i]: i += 1
            if i == istop: break
            istart.append(i)
            while i < istop and not schmitt_low[i]: i += 1
            if i == istop: break
            iend.append(i)  
        if len(istart) > len(iend): istart.pop()  #remove possible incomplete burst

        if plot:
            # plot detected regions with a line
            for ii in zip(istart, iend):
                plt.plot(np.array(ii) / self._sample_rate, [self._burst_trigger, self._burst_trigger], 'k-')

        # find frequency in bursts and validate
        frequency = []
        arrival_time = []
        transit_time = []
        validated = []
        
        f = np.arange(self._nw) / (self._nw * dt)
        #ipick = int(np.round(0.386837436/ dt)) # select a burst spectrum to plot
        if len(istart) > 0:
            ipick = istart[1]
        else:
            ipick = None

        for i, j in zip(istart, iend):
            if j-i > self._nw: break  # burst too long to handle
            window = np.zeros(self._nw)
            window[0:j-i] = sigfilt[i:j]
            sigf = np.fft.fft(window)
            sigspec = np.real(sigf * sigf.conjugate()) / (self._nw*dt)
            ipeak = np.argmax(sigspec)
            ifrac = self._gauss_interpolate(sigspec[ipeak-1:ipeak+2])
            freq = (ipeak + ifrac) / (self._nw * dt)
            transit = (j - i) * dt
            nfringes = freq * transit
            frequency.append(freq)
            arrival_time.append(i * dt)
            transit_time.append(transit)
            validated.append(self._min_periods <= nfringes and nfringes < self._max_periods)
            
            if plot and i == ipick:
                plt.figure(2)
                plt.clf()
                plt.plot(f[1:self._nw//2], sigspec[1:self._nw//2], '-o')
                
                velocity = (freq - self._optical_shift) / self._fdcal
                plt.title("t = %9.7f s, velocity = %6.3f m/s" %
                        (i * dt, velocity))

        # write validated velocities to file
        proc_f = os.path.join(self._proc_data_path, filename)
        with open(proc_f + '.txt', 'w') as f:
            f.write('# arrival time [s]   velocity [m/s]   transit time [s]\n')
            if len(validated) == 0:
                validationrate = 0
            else:
                validationrate = 100 * np.count_nonzero(validated) / len(validated)
            f.write('# validation rate: {:.1f}%\n'.format(validationrate))
            for i in range(len(frequency)):
                if validated[i]:
                    velocity = (frequency[i] - self._optical_shift ) / self._fdcal
                    f.write('{:.9f}  {:.3f}  {:.7f}\n'.format( 
                            arrival_time[i], velocity, transit_time[i]))

    def eval_all_files(self, plot_index: int = None):
        """ Evaluate all files in data_path"""
        for i, filename in enumerate(self._filename):
            print('Evaluating file {} of {}'.format(i+1, len(self._filename)))
            if i == plot_index:
                plot = True
            self.eval_file(filename, plot=plot)    
                    
if __name__ == "__main__":
    # parent_dir = os.path.dirname(os.getcwd())
    # data_path = os.path.join(parent_dir, r'LDA_DATA\20240115-0001_test2_higher level of signal')
    # proc_data_path = os.path.join(parent_dir, r'LDA_DATA\proc_20240115-0001_test2_higher level of signal')
    data_path = r'LDA'
    proc_data_path = r'LDA'
    lda = LDA_calcs(data_path, proc_data_path)
    lda.eval_all_files(plot_index=0)
    plt.show()
                
