# -*- coding: utf-8 -*-
"""
CTA measurement med NI DAQ

Author: Knud Erik Meyer 2020
Modified: Simon Lautrup Ribergaard 20210104 - simplified for EFM
          Benjamin Arnold Krekeler Hartz 20230102 - Added data buffer 
"""

import nidaqmx
from nidaqmx.constants import AcquisitionType
from nidaqmx.stream_readers import AnalogSingleChannelReader
#import os
#import glob
import numpy as np
import matplotlib.pyplot as plt


def getdata(datarate, nsample, channel):
    """Sample nsample points with datarate - return data as array"""
    data = np.zeros(nsample)
    timeout = nsample/datarate + 3
    with nidaqmx.Task() as task:
        task.ai_channels.add_ai_voltage_chan(channel)
        task.timing.cfg_samp_clk_timing(datarate,
                                        sample_mode=AcquisitionType.CONTINUOUS)
        # Buffer estimation
        minbufsize = 200
        fusb = 30
        
        # Calculation of the buffer size
        bufsize = int(nsample-round(nsample/fusb)) + minbufsize
        
        # Lowester buffer size check
        if round(nsample/fusb)<1:
            bufsize = minbufsize
          
        task.in_stream.input_buf_size = bufsize
        reader = AnalogSingleChannelReader(task.in_stream)
        reader.read_many_sample(data, nsample, timeout)
    return data
        
if __name__ == "__main__":
    
    # Set measurement channel (use NI MAX program to identify name)
    # Typically the name is "cDAQ<number>Mod1/ai0" where <number> may change
    channel = "cDAQ1Mod1/ai0"

    # Set sample frequency, fs
    fs = 600
    # Set number of samples, nsam
    nsam = fs*2
    
    # Record data as an array of voltages
    data_out = getdata(fs,nsam, channel)

    # Plot time series
    time_out = np.linspace(0,nsam-1,nsam)*(1/fs) # Creates time vector
    plt.plot(time_out,data_out)
    plt.xlim((time_out[0],time_out[-1]))
    plt.ylim((0,np.max(data_out)))
    plt.xlabel("Time [s]")

# Save data
Data_export = np.stack((time_out, data_out), axis=1) #Numpy array with time in first collumn and voltage in second collumn
np.savetxt('Hot-wire/Measurement/Data/Profile1/y=60_P=59,30.csv', Data_export, delimiter=',')