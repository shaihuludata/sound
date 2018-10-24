import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

wav_in_file = "solo_d_04.wav"
filename = wav_in_file.split(".")[0]
wav_dir = "./input/"
res_dir = "./result/"

wav_in_file = "{}{}".format(wav_dir, wav_in_file)
png_in = "{}{}{}".format(res_dir, filename, ".png")

sample_rate, samples = wavfile.read(wav_in_file)
frequencies, times, spectrogram = signal.spectrogram(samples, sample_rate)

fig = plt.figure(figsize=(20,10))
plt.subplot(321)
plt.plot(samples)
plt.subplot(322)
plt.plot(frequencies, spectrogram)
#plt.pcolormesh(times, frequencies, spectrogram)
#plt.imshow(spectrogram)
#plt.ylabel('Frequency [Hz]')
#plt.xlabel('Time [sec]')
#plt.show()




import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Filter requirements.
order = 6
fs = 30.0       # sample rate, Hz
cutoff = 3.667  # desired cutoff frequency of the filter, Hz

# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

# Plot the frequency response.
w, h = freqz(b, a, worN=8000)
plt.subplot(324)
plt.plot(0.5*fs*w/np.pi, np.abs(h))
# plt.plot(cutoff, 0.5*np.sqrt(2))
# plt.axvline(cutoff, color='k')
# plt.xlim(0, 0.5*fs)
# plt.title("Lowpass Filter Frequency Response")
# plt.xlabel('Frequency [Hz]')
# plt.grid()
plt.savefig(png_in)

# # Demonstrate the use of the filter.
# # First make some data to be filtered.
# T = 5.0         # seconds
# n = int(T * fs) # total number of samples
# t = np.linspace(0, T, n, endpoint=False)
# # "Noisy" data.  We want to recover the 1.2 Hz signal from this.
# data = np.sin(1.2*2*np.pi*t) + 1.5*np.cos(9*2*np.pi*t) + 0.5*np.sin(12.0*2*np.pi*t)
#
# # Filter the data, and plot both the original and filtered signals.
# y = butter_lowpass_filter(data, cutoff, fs, order)
#
# plt.subplot(2, 1, 2)
# plt.plot(t, data, 'b-', label='data')
# plt.plot(t, y, 'g-', linewidth=2, label='filtered data')
# plt.xlabel('Time [sec]')
# plt.grid()
# plt.legend()
#
# plt.subplots_adjust(hspace=0.35)
# plt.show()