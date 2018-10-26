import numpy as np
import matplotlib.pyplot as plt
import math
pi = math.pi

def export_formant_spectre(spectrogram, npy_spectr):
    num_of_samples = spectrogram.shape[0]
    formant_spectre2 = np.zeros(num_of_samples)
    for sample_spectre in np.transpose(spectrogram):
        formant_spectre2 += (sample_spectre / np.max(sample_spectre))**2
    formant_spectre = np.sqrt(formant_spectre2)/2  # / num_of_samples)
    # for sample_spectre in np.transpose(spectrogram):
    #     formant_spectre2 += (sample_spectre / np.max(sample_spectre))
    # formant_spectre = formant_spectre2 / 20 # /num_of_samples
    np.save(npy_spectr, formant_spectre)
    return formant_spectre


def make_filter_from_voice(fil_dir, filename):
    file_npy = "{}{}{}".format(fil_dir, filename, ".npy")
    H_w = np.load(file_npy)
    h_t = np.fft.ifft(H_w)
    h_t_out = "{}{}{}{}".format(fil_dir, "ht_", filename, ".npy")
    np.save(h_t_out, h_t)

    # fig = plt.figure(figsize=(20,10))
    # plt.subplot(211)
    # plt.plot(h_t)
    # plt.subplot(212)
    # plt.plot(H_w)
    # plt.savefig("./filter.png")
    return h_t_out
