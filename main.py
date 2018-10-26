import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from scipy.io import wavfile
from voice_filter import export_formant_spectre, make_filter_from_voice

wav_dir = "./input/"
res_dir = "./result/"
fil_dir = "./filters/"

# sox solo_d.wav -r 4000 solo_d_04.wav
#wav_in = "solo_d_08.wav"
# wav_in = "solo_d.wav"
# wav_in = "ciplyatki.wav"
wav_v_source = "ciplyatki_04.wav"
# wav_in = "duo_dn.wav"
wav_v_noise = "ciplyatki_04.wav"
#wav_v_noise = "ciplyatki_circular_04.wav"

v_filename = wav_v_source.split(".")[0]
v_noise_filename = wav_v_noise.split(".")[0]

wav_voice_in = "{}{}{}".format(wav_dir, v_filename, ".wav")
wav_voice_out= "{}{}{}".format(wav_dir, v_filename, "_out.wav")
png_file = "{}{}{}".format(res_dir, v_noise_filename, ".png")
npy_v_spectr = "{}{}{}".format(fil_dir, v_filename, ".npy")
wav_v_noise_in = "{}{}{}".format(wav_dir, v_noise_filename, ".wav")
wav_v_noise_out = "{}{}{}".format(wav_dir, v_noise_filename, "_out.wav")

# Working with voice wav file ##################################
sample_rate_v, samples_v = wavfile.read(wav_voice_in)
samples_arange_v = np.arange(0, samples_v.shape[0]/sample_rate_v, 1/sample_rate_v)
frequencies_v, times, spectrogram_v = sig.spectrogram(samples_v, sample_rate_v)
formant_spectre = export_formant_spectre(spectrogram_v, npy_v_spectr)

# Get filter
h_t_filename = make_filter_from_voice(fil_dir, v_filename)
h_t = np.load(h_t_filename).astype(float)
H_w = np.fft.fft(h_t)

# Working with noised wav file ##################################
sample_rate_n, samples_n_in = wavfile.read(wav_v_noise_in)
samples_arange_n_in = np.arange(0, samples_n_in.shape[0]/sample_rate_n, 1/sample_rate_n)
frequencies_n_in, times_n, spectrogram_n_in = sig.spectrogram(samples_n_in, sample_rate_n)

# Apply filter ########################################
samples_n_out = np.convolve(samples_n_in, h_t)
samples_arange_n_out = np.arange(0, samples_n_out.shape[0]/sample_rate_n, 1/sample_rate_n)

_, _, spectrogram_n_out = sig.spectrogram(samples_n_out, sample_rate_n)
# freq_out = np.arange(0, samples_out.shape[0]/sample_rate, freq_step)

# Plot it all #############################################
fig = plt.figure(figsize=(20,10))
plt.subplot(521)
plt.plot(samples_arange_v, samples_v)
plt.subplot(522)
plt.plot(frequencies_v, spectrogram_v)
# plt.pcolormesh(times, frequencies, spectrogram)
# plt.imshow(spectrogram)
# plt.ylabel('Frequency [Hz]')
# plt.xlabel('Time [sec]')
# plt.show()
plt.subplot(523)

plt.subplot(524)
plt.plot(frequencies_v, formant_spectre)
plt.subplot(525)
plt.plot(h_t)
plt.subplot(526)
plt.plot(frequencies_v, H_w)
# plt.plot(cutoff, 0.5*np.sqrt(2))
# plt.axvline(cutoff, color='k')
# plt.xlim(0, 0.5*fs)
# plt.title("Lowpass Filter Frequency Response")
# plt.xlabel('Frequency [Hz]')
# plt.grid()
plt.subplot(527)
plt.plot(samples_arange_n_in, samples_n_in)
plt.subplot(528)
plt.plot(frequencies_n_in, spectrogram_n_in)
plt.subplot(5, 2, 9)
plt.plot(samples_arange_n_out, samples_n_out)
plt.subplot(5, 2, 10)
plt.plot(frequencies_n_in, spectrogram_n_out)
# plt.xlabel('Time [sec]')
# plt.grid()
# plt.legend()
# plt.plot(frequencies, spectrogram_out)
# plt.subplots_adjust(hspace=0.35)
# plt.show()
plt.savefig(png_file)
print("{} saved".format(png_file))

# Save output wav #########################################
wavfile.write(wav_v_noise_out, sample_rate_n, samples_n_out)
print("{} saved".format(wav_v_noise_out))
