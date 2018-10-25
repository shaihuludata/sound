import numpy as np
import matplotlib.pyplot as plt


# fil = np.ndarray([0.1, -2, 0.3, -4, 0.5])

fil = [2, 1.8, 1.5, 1, 0.5, 0.2, 0.1, 0.1, 0, 0, 0, 0]

H_w = np.fft.fft(fil)
print(H_w)

fig = plt.figure(figsize=(20,10))
plt.subplot(211)
plt.plot(fil)
plt.subplot(212)
plt.plot(H_w)

plt.savefig("./filter.png")
