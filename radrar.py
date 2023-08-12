# -*- coding: utf-8 -*-
"""
Created on Sun Aug  6 09:41:53 2023

@author: batma
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt
from scipy.fft import fft, fftfreq,fft2
from scipy.io import wavfile

file = r"C:\Users\batma\Downloads\sample-3s.wav"
sr,data = wavfile.read(file)
print(f"number of channels = {data.shape[1]}")
print(data.size)

# Calculate the FFT of the data
amp1 = fft2(data)
print(amp1.shape)
freq = fftfreq(data.shape[0], d=1/sr) # Calculate frequencies
print(freq.shape)

# Design a low-pass Butterworth filter
cutoff_freq = 1000
order = 4
sos = butter(order, cutoff_freq, btype='low', fs=sr, output='sos')

# Apply the filter to the FFT data
filtered_stft = sosfilt(sos, amp1, axis=0)
print(filtered_stft.shape)

# Plot the filtered spectrum
fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)
img = ax[0].plot(freq, np.abs(amp1))  # Use np.abs to plot the magnitude
ax[0].set(title='unFiltered Spectrum', xlabel='Frequency (Hz)', ylabel='Magnitude')
ax[0].label_outer()
img = ax[1].plot(freq, np.abs(filtered_stft))  # Use np.abs to plot the magnitude
ax[1].set(title='Filtered Spectrum', xlabel='Frequency (Hz)', ylabel='Magnitude')
ax[1].label_outer()
plt.show()
