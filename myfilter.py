import numpy as np
import matplotlib.pyplot as plt
import math

pi = math.pi

fs = 44000
f_cutoff = 100
fc_norm = 2*pi*f_cutoff/fs

# fil = np.ndarray([0.1, -2, 0.3, -4, 0.5])
fil_array = range(-10, 11)
fil = list()
for t in fil_array:
    try:
        fil.append(math.sin(fc_norm * t) / (pi * t))
    except:
        fil.append(fc_norm/pi)

print(fil)
np.save("./filters/filter.npy", fil)
# fil = [2, 1.8, 1.5, 1, 0.5, 0.2, 0.1, 0.1, 0, 0, 0, 0]

H_w = np.fft.fft(fil)

fig = plt.figure(figsize=(20,10))
plt.subplot(211)
plt.plot(fil)
plt.subplot(212)
plt.plot(H_w)

plt.savefig("./filter.png")
