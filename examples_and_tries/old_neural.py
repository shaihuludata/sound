import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import wavfile


def neural_optimization(np_fir_filter: np.ndarray,
                        np_v_signal: np.ndarray,
                        np_n_signal: np.ndarray):
    shape_d = np_v_signal.shape[0]
    tf_fir_filter = tf.get_variable(name="fir", shape=[1, 26, 1])
    print(tf_fir_filter, np_fir_filter.shape)
    tf_v_signal = tf.placeholder(tf.float32, shape=[1, shape_d, 1], name="sig_v")
    print(tf_v_signal, np_v_signal.shape)
    tf_n_signal = tf.placeholder(tf.float32, shape=[1, shape_d, 1], name="sig_n")
    print(tf_n_signal, np_n_signal.shape)
    print(tf_fir_filter.shape, tf_n_signal.shape, tf_v_signal.shape)
    out_signal = tf.nn.conv1d(np_n_signal.reshape(13568, 1, shape_d//13568), tf_fir_filter, 1, "VALID")
    # out_signal = tf.nn.conv2d(tf_n_signal, tf_fir_filter, strides=[1, 1, 12, 1], padding="SAME")
    # out_signal = tf.nn.convolution(tf_n_signal, tf_fir_filter, "SAME", data_format="NC")
    loss = tf.reduce_mean(tf.square(tf.subtract(tf_v_signal, out_signal)))
    # метод оптимизации пока пусть будет такой
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    losses = dict()
    for i in range(0, 1):
        with tf.Session() as session:
            tf.global_variables_initializer().run()
            feed_dict = {tf_n_signal: [np_n_signal.reshape(shape_d, 1)],
                         tf_v_signal: [np_v_signal.reshape(shape_d, 1)]}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            losses[l] = (tf_fir_filter.eval(), tf.subtract(tf_v_signal, out_signal))
        print(i, end=' ')
    print('\n', losses.keys())
    best_result = min(list(losses.keys()))
    best = losses[best_result][0][0].transpose()[0]
    loses = losses[best_result][1]  # [0].transpose()[0]
    print(best_result, best, loses)
    return best


wav_in = "ciplyatki.wav"
wav_dir = "../input/"
wav_in = "{}{}".format(wav_dir, wav_in)
sample_rate, signal_v = wavfile.read(wav_in)


wav_in = "ciplyatki_noize.wav"
wav_dir = "../input/"
wav_in = "{}{}".format(wav_dir, wav_in)
sample_rate, signal_n = wavfile.read(wav_in)

# fir_filt = np.array([0.1, 0.2, 0.5, 0.8, 1.0, 1.0, 0.8, 0.5, 0.2, 0.1, 0.0, 0.0], dtype='float32')
fir_filt = np.array([0.004530021997303397, 0.004532951759520995, 0.004535574088523477, 0.004537888663561965, 0.004539895201512734, 0.0045415934569184475, 0.0045429832220239075, 0.004544064326806308, 0.0045448366390000005, 0.00454530006411576, 0.004545454545454546, 0.00454530006411576, 0.0045448366390000005, 0.004544064326806308, 0.0045429832220239075, 0.0045415934569184475, 0.004539895201512734, 0.004537888663561965, 0.004535574088523477, 0.004532951759520995, 0.004530021997303397], dtype='float32')
# signal_v = np.array([1.0, 2.0, 3.0, 3.0, 3.5, 3.0, 3.0, 2.5, 2.0, 1.0, 1.0, 1.0], dtype='float32')
# , 1.0, 2.0, 3.0, 8.0, 4.0, 2.0, 3.0, 4.0, 6.0, 2.0, 4.0, 6.0], dtype='float32')
# signal_n = np.array([1.0, 2.0, 3.5, 3.0, 3.5, 3.0, 3.0, 2.5, 2.0, 1.0, 1.5, 1.0], dtype='float32')
# , 1.0, 2.5, 3.0, 8.0, 4.0, 2.5, 3.0, 4.0, 6.5, 2.0, 4.5, 6.0], dtype='float32')

best_filter = neural_optimization(fir_filt, signal_v, signal_n)
print(best_filter)

out_sig = np.convolve(signal_n, best_filter)
# print(out_sig)

fig = plt.figure(figsize=(30, 10))
plt.subplot(411)
plt.plot(signal_v)
plt.subplot(412)
plt.plot(signal_n)
plt.subplot(413)
plt.plot(out_sig)
plt.subplot(414)
plt.plot(best_filter)
plt.show()
