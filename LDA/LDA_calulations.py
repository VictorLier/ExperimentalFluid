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
        self._gain = 100.0
        self._filter_center = 1.45e6
        self._filter_width = 0.5e6
        self._min_periods = 15
        self._max_periods = 100
        self._win_width_Hilbert = 200
        self._burst_trigger = 1.0
        self._burst_schmitt = 0.1
        self._gaussfit_width = 4

        # Filter meathod
        #self._filter_method = 'butter'
        self._filter_method = 'highpass'

        # filter signal and find envelope
        # self._band = [self._filter_center - 0.5 * self._filter_width, self._filter_center + 0.5 * self._filter_width]
        self._band = [np.inf, 200000]

        self._nw = 2048
        self._dt = 1 / self._sample_rate

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

    def _highpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs
        cut = cutoff / nyq
        b, a = butter(order, cut, btype='hp')
        return b, a

    def _highpass_filter(self, data, cutoff, fs, order=5):
        b, a = self._highpass(cutoff, fs, order=order)
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
        sig1 = data1['A'][0,:] 
        sig2 = np.nan_to_num(sig1, copy=False, neginf=0) # remove -inf in data
        sig = self._gain * sig2
        self._dt = 1 / self._sample_rate
        times = np.arange(len(sig)) * self._dt

        if self._filter_method == 'butter':
            sigfilt = self._butter_bandpass_filter(sig, self._band[0], self._band[1], self._sample_rate)
        elif self._filter_method == 'highpass':
            sigfilt = self._highpass_filter(sig, self._band[1], self._sample_rate)
        else:
            raise ValueError('Filter method not recognized')
        
        sig_hilbert = np.abs(hilbert(sigfilt))
        sig_envelope = np.convolve(sig_hilbert, 
                                    np.ones(self._win_width_Hilbert)/self._win_width_Hilbert,
                                    mode='same')
        
        if plot:
            plt.figure()
            plt.title('Raw signal')
            plt.plot(times, sig)
            plt.xlabel('Time [s]')
            plt.ylabel('Signal')

            plt.figure()
            plt.title('Hilbert')
            plt.plot(times, sigfilt)
            plt.plot(times, sig_hilbert)
            plt.xlabel('Time [s]')
            plt.ylabel('Signal')


            plt.figure()
            plt.title('Filtered signal and envelope')
            plt.plot(times, sigfilt)
            plt.plot(times, sig_envelope)
            plt.xlabel('Time [s]')
            plt.ylabel('Signal')

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
            plt.figure()
            plt.title('Enveloped signal with burst detection')
            plt.plot(times, sigfilt)
            plt.plot(times, sig_envelope)
            plt.xlabel('Time [s]')
            plt.ylabel('Signal')
            for ii in zip(istart, iend):
                plt.plot(np.array(ii) / self._sample_rate, [self._burst_trigger, self._burst_trigger], 'k-')

        # find frequency in bursts and validate
        frequency = []
        arrival_time = []
        transit_time = []
        validated = []
        
        f = np.arange(self._nw) / (self._nw * self._dt)
        #ipick = int(np.round(0.386837436/ self._dt)) # select a burst spectrum to plot
        if len(istart) > 0:
            ipick = istart[30]
        else:
            ipick = None

        for i, j in zip(istart, iend):
            if j-i > self._nw:
                continue  # burst too long to handle
            window = np.zeros(self._nw)
            window[0:j-i] = sigfilt[i:j]
            sigf = np.fft.fft(window)
            sigspec = np.real(sigf * sigf.conjugate()) / (self._nw*self._dt)
            # ipeak = np.argmax(sigspec) # Old method: Did not work beacause of symmetric spectrum - sometimes the peak was chosen to be the wrong side of the spectrum
            ipeak = np.argmax(sigspec[0:self._nw//2])
            ifrac = self._gauss_interpolate(sigspec[ipeak-1:ipeak+2])
            freq = (ipeak + ifrac) / (self._nw * self._dt)
            transit = (j - i) * self._dt
            nfringes = freq * transit
            frequency.append(freq)
            arrival_time.append(i * self._dt)
            transit_time.append(transit)
            validated.append(self._min_periods <= nfringes and nfringes < self._max_periods)
            
            velocity = (freq - self._optical_shift) / self._fdcal
            #print(j-i, ipeak, velocity, freq)
            #if plot and i == ipick:
            if plot and i == ipick:
            #if velocity > 2:
                print('velocity: {:.3f} m/s, nfringes: {:.3f}, freq: {:.3f} Hz, transit: {:.8f} s, ipeak: {}, ifrac: {}'.format(velocity, nfringes, freq, transit, ipeak, ifrac))
                plt.figure()
                plt.plot(f[1:self._nw//2], sigspec[1:self._nw//2], '-o')
                #plt.plot(f, sigspec, '-o')
                # Indicate the peak
                plt.plot(f[ipeak], sigspec[ipeak], 'ro')
                velocity = (freq - self._optical_shift) / self._fdcal
                plt.title("t = %9.7f s, velocity = %6.3f m/s, nfringes:  %f, freq: %f, transit: %f, ipeak: %i, ifrac: %f" %
                        (i * self._dt, velocity, nfringes, freq, transit, ipeak, ifrac))

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
                    f.write('{:.9f}  {}  {:.7f}\n'.format( 
                            arrival_time[i], velocity, transit_time[i]))

    def eval_all_files(self, plot_index: int = None):
        """ Evaluate all files in data_path"""
        print('Evaluating all files in {}'.format(self._data_path))
        for i, filename in enumerate(self._filename):
            print('Evaluating file {} of {}'.format(i+1, len(self._filename)))
            plot = i == plot_index
            self.eval_file(filename, plot=plot)    

    def plot_all_raw_files(self, plot_index: int = None):
        # Get all velocity data
        velocity = []
        for i, filename in enumerate(self._filename):
            print('Evaluating file {} of {}'.format(i+1, len(self._filename)))
            data_f = os.path.join(self._proc_data_path, filename)
            data1 = np.loadtxt(data_f+'.txt', skiprows=2)
            velocity.append(list(data1[:,1]))
        
        velocity_total = [item for row in velocity for item in row]
        plt.figure()
        plt.title('Histogram of all velocities measurements')
        plt.hist(velocity_total, bins=100)
        plt.xlabel('Velocity [m/s]')
        plt.ylabel('Counts')
        plt.grid()

        if plot_index is not None:
            plt.figure()
            plt.title('Histogram of velocity for file {}'.format(self._filename[plot_index]))
            plt.hist(velocity[plot_index], bins=len(velocity[plot_index])//3)
            plt.xlabel('Counts')
            plt.ylabel('Velocity [m/s]')
            plt.grid()

    def remove_outliers(self, filename: str):
        """ Remove outliers from data file"""
        #get signal data from data file
        data_f = os.path.join(self._proc_data_path, filename)
        data1 = np.loadtxt(data_f+'.txt', skiprows=2)
        arrival_time = np.array(data1[:,0])
        velocity = np.array(data1[:,1])
        transit_time = np.array(data1[:,2])

        # Calculate mean and weigth according to transit time
        mean_velocity = np.average(velocity, weights=transit_time)

        # Calculate standard deviation from weighted mean
        std_velocity = np.sqrt(np.sum((velocity-mean_velocity)**2)/(len(velocity)-1))

        # Make valid mask
        valid_mask = np.abs(velocity - mean_velocity) < 3*std_velocity

        # Write valid data to file
        with open(data_f + '_valid.txt', 'w') as f:
            f.write('# arrival time [s]   velocity [m/s]   transit time [s]\n')
            for i in range(len(valid_mask)):
                if valid_mask[i]:
                    f.write('{:.9f}  {}  {:.7f}\n'.format( 
                            arrival_time[i], velocity[i], transit_time[i]))

        stop = True

    def remove_outliers_all_files(self):
        """ Remove outliers from all data files"""
        for i, filename in enumerate(self._filename):
            print('Evaluating file {} of {}'.format(i+1, len(self._filename)))
            self.remove_outliers(filename)
    
    def get_stats(self, filename: str):
        """ Get mean velocity from data file"""
        data_f = os.path.join(self._proc_data_path, filename)
        data1 = np.loadtxt(data_f+'_valid.txt', skiprows=1)
        arrival_time = np.array(data1[:,0])
        velocity = np.array(data1[:,1])
        transit_time = np.array(data1[:,2])

        # Calculate mean and weigth according to transit time
        mean_velocity = np.average(velocity, weights=transit_time)

        # Calculate standard deviation and weigth according to transit time
        std_velocity = np.sqrt(np.sum((velocity-mean_velocity)**2)/(len(velocity)-1))

        return mean_velocity, std_velocity
    
    def get_stats_all_files(self):

        """ Get mean velocity from all data files"""
        mean_velocity = []
        std_velocity = []
        for i, filename in enumerate(self._filename):
            mean, std = self.get_stats(filename)
            mean_velocity.append(mean)
            std_velocity.append(std)
        
        # save to file
        data_f = os.path.join(self._proc_data_path, 'stats.txt')
        with open(data_f, 'w') as f:
            f.write('# mean velocity [m/s]   std velocity [m/s]\n')
            for i in range(len(mean_velocity)):
                f.write('{}  {}\n'.format(mean_velocity[i], std_velocity[i]))

    def get_mean_velocity(self):
        # First version mean velocity from stats.txt
        # data_f = os.path.join(self._proc_data_path, 'stats.txt')
        # data1 = np.loadtxt(data_f, skiprows=1)
        # mean_v = np.array(data1[:,0])
        # std_v = np.array(data1[:,1])
        # mean_velocity_1 = np.mean(mean_v)

        # Alternative version mean velocity from all data files
        # get all valid velocities and transit times
        velocity = []
        transit_time = []
        for i, filename in enumerate(self._filename):
            data_f = os.path.join(self._proc_data_path, filename)
            data1 = np.loadtxt(data_f+'_valid.txt', skiprows=1)
            velocity.append(list(data1[:,1]))
            transit_time.append(list(data1[:,2]))
        
        velocity_total = [item for row in velocity for item in row]
        transit_time_total = [item for row in transit_time for item in row]
        mean_velocity_2 = np.average(velocity_total, weights=transit_time_total)
        std_velocity_2 = np.sqrt(np.sum((velocity_total-mean_velocity_2)**2)/(len(velocity_total)-1))

        return mean_velocity_2, std_velocity_2

    def peak_to_velocity_plot(self):
        velocity = []
        for ipeak in range(self._nw):
            freq = (ipeak) / (self._nw * self._dt)
            v = (freq - self._optical_shift) / self._fdcal
            velocity.append(v)
        plt.figure()
        plt.title('Peak to velocity')
        plt.plot(range(self._nw), velocity)
        plt.xlabel('Peak')
        plt.ylabel('Velocity [m/s]')
        plt.grid()
        plt.show()

    def full_evaluation(self, plot_index: int = None):
        """ Evaluate all files in data_path"""
        self.eval_all_files(plot_index)
        self.remove_outliers_all_files()
        self.get_stats_all_files()
    
if __name__ == "__main__":
    parent_dir = os.path.dirname(os.getcwd())
    # data_path = os.path.join(parent_dir, r'LDA_DATA_OLD\20240116-0001_test2')
    # proc_data_path = os.path.join(parent_dir, r'LDA_DATA_OLD\proc_20240116-0001_test2')
    # data_path = os.path.join(parent_dir, r'LDA_DATA_OLD\20240116-0001_7_0cm')
    # proc_data_path = os.path.join(parent_dir, r'LDA_DATA_OLD\proc_20240116-0001_7_0cm')
    data_path = r'LDA'
    proc_data_path = r'LDA'
    # data_path = os.path.join(parent_dir, r'LDA_DATA\20240117-0001_test3_slowjet')
    # proc_data_path = os.path.join(parent_dir, r'LDA_DATA\20240117-0001_test3_slowjet')
    lda = LDA_calcs(data_path, proc_data_path)
    
    #lda.peak_to_velocity_plot()

    if True:
        # Eval data
        lda.eval_file('20231213-100_015_0001_02', plot=True)
        # lda.eval_all_files(plot_index=7)
        # lda.remove_outliers_all_files()
        # lda.get_stats_all_files()
        # lda.full_evaluation()
    
    if True:
        # Visualize data
        lda.plot_all_raw_files(plot_index=0)
    
    if False:
        # Get mean velocity
        lda.get_mean_velocity()
        stop = True


    plt.show()
                
