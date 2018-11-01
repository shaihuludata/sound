import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


def neural_optimization(np_fir_filter: np.ndarray,
                        np_v_signal: np.ndarray,
                        np_n_signal: np.ndarray):
    tf_fir_filter = tf.get_variable(name="fir", shape=[1, 12, 1])
    print(tf_fir_filter, np_fir_filter.shape)
    tf_v_signal = tf.placeholder(tf.float32, shape=[1, 12, 1], name="sig_v")
    print(tf_v_signal, np_v_signal.shape)
    tf_n_signal = tf.placeholder(tf.float32, shape=[1, 12, 1], name="sig_n")
    print(tf_n_signal, np_n_signal.shape)
    out_signal = tf.nn.conv1d(np_n_signal.reshape(1, 1, 12), tf_fir_filter, 1, "SAME")
    # out_signal = tf.nn.conv2d(tf_n_signal, tf_fir_filter, strides=[1, 1, 12, 1], padding="SAME")
    # out_signal = tf.nn.convolution(tf_n_signal, tf_fir_filter, "SAME", data_format="NC")
    loss = tf.reduce_mean(tf.square(tf.subtract(tf_v_signal, out_signal)))
    # метод оптимизации пока пусть будет такой
    optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)
    losses = dict()
    for i in range(0, 500):
        with tf.Session() as session:
            tf.global_variables_initializer().run()
            feed_dict = {tf_n_signal: [np_n_signal.reshape(12, 1)],
                         tf_v_signal: [np_v_signal.reshape(12, 1)]}
            _, l = session.run([optimizer, loss], feed_dict=feed_dict)
            losses[l] = tf_fir_filter.eval()
        print(i)
    print([i for i in losses])
    best_result = min(list(losses.keys()))
    best = losses[best_result][0].transpose()[0]
    print(best)
    return best


fir_filter = np.array([0.1, 0.2, 0.5, 0.8, 1.0, 1.0, 0.8, 0.5, 0.2, 0.1, 0.0, 0.0], dtype='float32')
signal_v = np.array([1.0, 2.0, 3.0, 8.0, 4.0, 2.0, 3.0, 4.0, 6.0, 2.0, 4.0, 6.0], dtype='float32')
signal_n = np.array([1.0, 2.5, 3.0, 8.0, 4.0, 2.5, 3.0, 4.0, 6.5, 2.0, 4.5, 6.0], dtype='float32')

best_filter = neural_optimization(fir_filter, signal_v, signal_n)
out_sig = np.convolve(signal_n, best_filter)
print(out_sig)
fig = plt.figure(figsize=(30,10))
plt.subplot(311)
plt.plot(signal_v)
plt.subplot(312)
plt.plot(signal_n)
plt.subplot(313)
plt.plot(out_sig)
plt.show()
