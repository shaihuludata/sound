import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from scipy.io import wavfile


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = sig.lfilter(b, a, data)
    return y


# sox solo_d.wav -r 4000 solo_d_04.wav
#wav_in = "solo_d_08.wav"
# wav_in = "solo_d32.wav"
# wav_in = "solo_d.wav"
# wav_in = "ciplyatki.wav"
#wav_in = "ciplyatki_04.wav"
wav_in = "ciplyatki_noize.wav"
#wav_in = "ciplyatki32.wav"
# wav_in = "duo_dn.wav"

filename = wav_in.split(".")[0]
wav_dir = "./input/"
res_dir = "./result/"
fil_dir = "./filters/"

wav_in = "{}{}".format(wav_dir, wav_in)
wav_out= "{}{}{}".format(wav_dir, filename, "_out.wav")
png_in = "{}{}{}".format(res_dir, filename, ".png")
spectr = "{}{}{}".format(fil_dir, filename, ".npy")
# Working with wav file ##################################
sample_rate, samples = wavfile.read(wav_in)
samples_arange = np.arange(0, samples.shape[0]/sample_rate, 1/sample_rate)
frequencies, times, spectrogram = sig.spectrogram(samples, sample_rate)

# np.save(spectr, spectrogram)
# formant_spectre = np.ones(spectrogram.shape[0])
formant_spectre = np.zeros(spectrogram.shape[0])

for sample_spectre in np.transpose(spectrogram):
    formant_spectre += sample_spectre / np.max(sample_spectre)
formant_spectre = formant_spectre / spectrogram.shape[0]
np.save(spectr, formant_spectre)
# Applying filter ########################################
# Filter requirements.
# order = 6
# fs = sample_rate  # 4000.0    # sample rate, Hz
# cutoff = 500  # 3.667  # desired cutoff frequency of the filter, Hz
# # Get the filter coefficients so we can check its frequency response.
# b, a = butter_lowpass(cutoff, fs, order)
# w, h = sig.freqz(b, a, worN=8000)
# H_w = 0.5*fs*w/np.pi

# Filter the data, and plot filtered signals.
#samples_out = butter_lowpass_filter(samples, cutoff, fs, order)
fil_name = "ht_ciplyatki_04"
filter_filename = "{}{}{}".format(fil_dir, fil_name, ".npy")
h_t = np.load(filter_filename).astype(float)
H_w = np.fft.fft(h_t)

samples_out = np.convolve(samples, h_t)
# samples_out = sig.lfilter(fil, [1], samples)
samples_arange_out = np.arange(0, samples_out.shape[0]/sample_rate, 1/sample_rate)

_, _, spectrogram_out = sig.spectrogram(samples_out, sample_rate)
#freq_out = np.arange(0, samples_out.shape[0]/sample_rate, freq_step)

# Plot it all #############################################
fig = plt.figure(figsize=(20,10))
plt.subplot(421)
plt.plot(samples_arange, samples)
plt.subplot(422)
plt.plot(frequencies, spectrogram)
# plt.pcolormesh(times, frequencies, spectrogram)
# plt.imshow(spectrogram)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
plt.subplot(423)
plt.plot(h_t)
plt.subplot(424)
plt.plot(H_w)
# plt.plot(cutoff, 0.5*np.sqrt(2))
# plt.axvline(cutoff, color='k')
# plt.xlim(0, 0.5*fs)
# plt.title("Lowpass Filter Frequency Response")
# plt.xlabel('Frequency [Hz]')
# plt.grid()
plt.subplot(425)
plt.plot(samples_arange_out, samples_out)  # , 'b-')
# plt.xlabel('Time [sec]')
# plt.grid()
# plt.legend()
plt.subplot(426)
#plt.plot(frequencies, spectrogram_out)
plt.plot(spectrogram_out)
# plt.subplots_adjust(hspace=0.35)
# plt.show()
plt.subplot(428)
plt.plot(frequencies, formant_spectre)

plt.savefig(png_in)

# Save output wav #########################################
wavfile.write(wav_out, sample_rate, samples_out)