from baseFunctions import *
import numpy as np

inp = tf.placeholder('float', [2, 5])
out = geebs_tanh(inp)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    matrix = np.array([[-1.7,  -1.1,   0.02,  1.63, -2.08], [1.56, -7.51, 2.84, 1.09, -3.74]]) #(np.random.randn(5, 5)).round(2)
    inp_, out_ = sess.run([inp, out], feed_dict={inp: matrix})
    print(np.round(inp_,3), "\n", np.round(out_,3))