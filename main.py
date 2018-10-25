import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from scipy.io import wavfile
from myfilter import fil
#from scipy.signal import butter, lfilter, freqz


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = sig.lfilter(b, a, data)
    return y


#sox solo_d.wav -r 4000 solo_d_04.wav
wav_in = "solo_d_04.wav"
filename = wav_in.split(".")[0]
wav_dir = "./input/"
res_dir = "./result/"

wav_in = "{}{}".format(wav_dir, wav_in)
wav_out= "{}{}{}".format(wav_dir, filename, "_out.wav")
png_in = "{}{}{}".format(res_dir, filename, ".png")

# Working with wav file ##################################
sample_rate, samples = wavfile.read(wav_in)
samples_arange = np.arange(0, samples.shape[0]/sample_rate, 1/sample_rate)
frequencies, times, spectrogram = sig.spectrogram(samples, sample_rate)

# Applying filter ########################################
# Filter requirements.
order = 6
fs = 4000.0    # sample rate, Hz
cutoff = 1000  # 3.667  # desired cutoff frequency of the filter, Hz
# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)
#
# w, h = sig.freqz(b, a, worN=8000)

# Filter the data, and plot filtered signals.
# samples_out = butter_lowpass_filter(samples, cutoff, fs, order)
print(a)
print(b)
samples_out = np.convolve(samples, fil)
samples_arange_out = np.arange(0, samples_out.shape[0]/sample_rate, 1/sample_rate)

_, _, spectrogram_out = sig.spectrogram(samples_out, sample_rate)

# Save output wav #########################################
wavfile.write(wav_out, sample_rate, samples_out)

# Plot it all     #########################################
fig = plt.figure(figsize=(20,10))
plt.subplot(321)
plt.plot(samples_arange, samples)
plt.subplot(322)
plt.plot(frequencies, spectrogram)
# plt.pcolormesh(times, frequencies, spectrogram)
# plt.imshow(spectrogram)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
plt.subplot(324)
#plt.plot(0.5*fs*w/np.pi, np.abs(h))
# plt.plot(cutoff, 0.5*np.sqrt(2))
# plt.axvline(cutoff, color='k')
# plt.xlim(0, 0.5*fs)
# plt.title("Lowpass Filter Frequency Response")
# plt.xlabel('Frequency [Hz]')
# plt.grid()
plt.subplot(325)
plt.plot(samples_arange_out, samples_out)  # , 'b-')
# plt.xlabel('Time [sec]')
# plt.grid()
# plt.legend()
plt.subplot(326)
plt.plot(frequencies, spectrogram_out)
# plt.subplots_adjust(hspace=0.35)
# plt.show()
plt.savefig(png_in)
