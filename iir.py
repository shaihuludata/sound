from scipy import signal as sig


def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = sig.lfilter(b, a, data)
    return y


# Filter requirements.
# order = 6
# fs = sample_rate  # 4000.0    # sample rate, Hz
# cutoff = 500  # 3.667  # desired cutoff frequency of the filter, Hz
# # Get the filter coefficients so we can check its frequency response.
# b, a = butter_lowpass(cutoff, fs, order)
# w, h = sig.freqz(b, a, worN=8000)
# H_w = 0.5*fs*w/np.pi

# samples_out = butter_lowpass_filter(samples, cutoff, fs, order)

# samples_out = sig.lfilter(fil, [1], samples)


