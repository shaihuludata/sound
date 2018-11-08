import numpy as np
from numpy.fft import fft
import tensorflow as tf
import matplotlib.pyplot as plt
import time

initial_filter_set = False
debug = False

if initial_filter_set:
    # np_fir_filter = np.array([0.1, 0.2, 0.5, 0.8, 1.0, 1.0, 0.8, 0.5, 0.2, 0.1], dtype='float32')
    np_fir_filter = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype='float32')
    fir_rank = np_fir_filter.shape[0]
    f_filter = tf.get_variable(name="fir", initializer=np_fir_filter.reshape([fir_rank, 1, 1]))
else:
    fir_rank = 10
    f_filter = tf.get_variable(name="fir", shape=[fir_rank, 1, 1])

trailer = np.zeros(fir_rank)
# trailer = np.zeros(0)
np_v_signal = np.array([1.0, 2.0, 3.0, 8.0, 4.0, 2.0, 3.0, 4.0, 6.0, 2.0, 4.0, 6.0], dtype='float32')
# np_n_signal = np.array([0.5, 2.5, 2.5, 8.5, 3.5, 2.5, 2.5, 4.5, 5.5, 2.5, 3.5, 6.5], dtype='float32')
np_n_signal = np.array([0.0, 3.0, 2.0, 9.0, 3.0, 3.0, 2.0, 5.0, 5.0, 3.0, 3.0, 7.0], dtype='float32')

np_v_signal = np.concatenate((trailer, np_v_signal, trailer))
np_n_signal = np.concatenate((trailer, np_n_signal, trailer))
sample_size = np_v_signal.shape[0]

v_signal = tf.placeholder(tf.float32, shape=[1, sample_size, 1], name="sig_v")
n_signal = tf.placeholder(tf.float32, shape=[1, sample_size, 1], name="sig_n")
f_signal = tf.nn.convolution(n_signal, f_filter, "SAME", name="sig_f")
# f_signal = tf.nn.conv1d(f_filter, n_signal, 12, "VALID", name="sig_f")

f_signal_l = tf.reshape(f_signal, [-1])
f_signal_l = tf.manip.roll(f_signal_l, -13, 0)  # -int(1.5*trailer.shape[0]), 0)
f_signal_l = tf.reshape(f_signal_l, (f_signal_l.shape[0]//sample_size, sample_size))

v_signal_l = tf.reshape(v_signal, [-1])
v_signal_l = tf.manip.roll(v_signal_l, -(trailer.shape[0]-1), 0)
v_signal_l = tf.reshape(v_signal_l, (v_signal_l.shape[0]//sample_size, sample_size))

if debug:
    print_f_sig = tf.Print(f_signal_l[0], [f_signal_l[0]])
    diff = tf.subtract(print_f_sig, v_signal_l[0], name="diff")
    loss = tf.reduce_mean(tf.square(diff), name="loss")
else:
    diff = tf.subtract(f_signal_l, v_signal_l, name="diff")
    loss = tf.reduce_mean(tf.square(diff), name="loss")

optimizer = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

summary_writer = tf.summary.FileWriter("./data", sess.graph)
summary_y = tf.summary.scalar('output', 1)


def teach_and_plot():
    losses = dict()

    feed_dict = {n_signal: np_n_signal.reshape(1, sample_size, 1),
                 v_signal: np_v_signal.reshape(1, sample_size, 1)}
    # feed_dict = {n_signal: [np_n_signal], v_signal: [np_v_signal]}

    for i in range(0, 100):
        with tf.Session() as session:
            tf.global_variables_initializer().run()
            opt, l, f_sig, dif, f_sig_l, v_sig_l = session.run([optimizer, loss, f_signal, diff, f_signal_l, v_signal_l], feed_dict=feed_dict)

            f_sig = f_sig.reshape([-1])
            summary_str = session.run(summary_y)
            summary_writer.add_summary(summary_str, i)

            cur_filt = f_filter.eval().reshape(fir_rank)
            # print(cur_filt)
            # print(opt)
            # print(f_sig)
            losses[l] = cur_filt, f_sig, dif, f_sig_l.reshape([32]), v_sig_l.reshape([32])
        print(i, l, opt)
    best_score = min(list(losses.keys()))
    best_result = losses[best_score]
    best_filter = best_result[0]  # .reshape(best_result.shape[1])
    best_f_sig = best_result[1]
    best_diff = best_result[2]
    best_f_sig_l = best_result[3]
    best_v_sig_l = best_result[4]
    # print(losses[best_score])
    # print(best, best.shape)

    out_sig = np.convolve(np_n_signal, best_filter)
    # out_sig *= (-1) if np.mean(out_sig) < 0 else 1
    best_f_sig *= (-1) if np.mean(best_f_sig) < 0 else 1
    # print(out_sig)

    fig = plt.figure(figsize=(30, 10))
    # ax = fig.add_subplot(5, 2, 9)
    plt.subplot(521)
    plt.title('исходный сигнал')
    plt.plot(np_v_signal)
    plt.subplot(522)
    plt.plot(fft(np_v_signal))
    plt.subplot(523)
    plt.title('зашумлённый сигнал')
    plt.plot(np_n_signal)
    plt.subplot(524)
    plt.plot(fft(np_n_signal))
    plt.subplot(525)
    plt.title('отфильтрованный сигнал')
    plt.plot(best_f_sig)
    # plt.plot(out_sig)
    plt.subplot(526)
    plt.plot(fft(best_f_sig))
    plt.subplot(527)
    plt.title('нормированный выходной и исходный')
    plt.plot(best_f_sig / max(best_f_sig))
    plt.plot(np_v_signal / max(np_v_signal))
    plt.subplot(528)
    fft_out_sig = fft(out_sig)
    fft_np_n_signal = fft(np_n_signal)
    plt.plot(fft_out_sig / max(fft_out_sig))
    plt.plot(fft_np_n_signal / max(fft_np_n_signal))
    plt.subplot(5, 2, 9)
    # plt.title('фильтр')
    # plt.plot(best_filter)
    plt.plot(best_f_sig_l)
    plt.plot(best_v_sig_l)
    plt.subplot(5, 2, 10)
    plt.plot(fft(best_filter))
    plt.savefig("./neural.png")
    plt.show()


teach_and_plot()
