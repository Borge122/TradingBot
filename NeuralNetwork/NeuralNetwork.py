import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from baseFunctions import *
from NN_layers import *
import pickle
dataset = pickle.load(open(r"C:\Users\Coding Projects\Desktop\Programming Projects\TradingBot\HistoricalRecords\Dataset.pickle", "rb"))
dataset = np.matmul(np.diag(1./np.max(np.abs(dataset), axis=1)), dataset)



def medium_extractor(alpha0, cellstates):
    # #[?, 20]
    cellstate_1, cellstate_2, cellstate_3, cellstate_4 = cellstates

    alpha1 = geebs_tanh(fcl(alpha0, [20, 64, 64],"a-FCL-1a"))
    alpha1, upper_cell1 = dense_lstm(alpha1, [128, 128, 64], "a-LSTM-1b", cellstate_1)

    alpha2 = geebs_tanh(fcl(alpha1, [64, 128, 128],"a-FCL-2a"))
    alpha2, upper_cell2 = dense_lstm(alpha2, [256, 256, 128], "a-LSTM-2b", cellstate_2)

    alpha3 = geebs_tanh(fcl(alpha2, [128, 256, 256],"a-FCL-3a"))
    alpha3, upper_cell3 = dense_lstm(alpha3, [512, 512, 256], "a-LSTM-3b", cellstate_3)

    beta0 = geebs_tanh(fcl(alpha0, [20, 256, 256, 128], "b-FCL-0a"))
    beta1 = geebs_tanh(fcl(alpha1, [64, 256, 256, 128], "b-FCL-1a"))
    beta2 = geebs_tanh(fcl(alpha2, [128, 256, 256, 128], "b-FCL-2a"))
    beta3 = geebs_tanh(fcl(alpha3, [256, 256, 256, 128], "b-FCL-3a"))

    attention = tf.nn.softmax(fcl(tf.concat([beta0, beta1, beta2, beta3], axis=1), [512, 256, 4], "Attention"))
    a0, a1, a2, a3 = tf.split(attention, num_or_size_splits=4, axis=1)

    alpha4 = geebs_tanh(fcl(tf.concat([tf.multiply(a0, beta0),
                               tf.multiply(a1, beta1),
                               tf.multiply(a2, beta2),
                               tf.multiply(a3, beta3)], axis=1), [512, 256, 128], "Attention"))
    output, upper_cell4 = geebs_tanh(fcl(dense_lstm(alpha4, [128, 512, 512], "a-LSTM-4b", cellstate_4), [512, 256, 256, 20], "FCL-Final"))

    cellstates = [[alpha1, upper_cell1], [alpha2, upper_cell2], [alpha3, upper_cell3], [output, upper_cell4]]

    return output, cellstates, attention

def trainer():
    input_batch = tf.placeholder('float', 20)
    label_batch = tf.placeholder('float', 20)

    cs1_lc = tf.placeholder('float', 64)
    cs1_uc = tf.placeholder('float', 64)
    cs2_lc = tf.placeholder('float', 128)
    cs2_uc = tf.placeholder('float', 128)
    cs3_lc = tf.placeholder('float', 256)
    cs3_uc = tf.placeholder('float', 256)
    cs4_lc = tf.placeholder('float', 512)
    cs4_uc = tf.placeholder('float', 512)

    with tf.variable_scope("TEST", reuse=tf.AUTO_REUSE):
        r_classification, cellstates, r_attention = medium_extractor(input_batch, [[cs1_lc, cs1_uc], [cs2_lc, cs2_uc], [cs3_lc, cs3_uc], [cs4_lc, cs4_uc]])
    r_cost = tf.math.reduce_mean(tf.math.reduce_sum(tf.multiply(0.5, tf.square(tf.subtract(label_batch, r_classification))), axis=1))
    r_train = tf.train.AdamOptimizer().minimize(r_cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        accuracy_list = []
        cstates = [[0,0], [0,0], [0,0], [0,0],]
        for i in range(dataset.shape[1]-1):
            if not i%10 == 0:
                dictionary = {input_batch: dataset[i], label_batch: dataset[i+1],
                              cs1_lc: cstates[0][0], cs1_uc: cstates[0][1],
                              cs2_lc: cstates[1][0], cs2_uc: cstates[1][1],
                              cs3_lc: cstates[2][0], cs3_uc: cstates[2][1],
                              cs4_lc: cstates[3][0], cs4_uc: cstates[3][1],}
                _, cstates = sess.run([r_train, cellstates], feed_dict=dictionary)
            else:
                dictionary = {input_batch: dataset[i], label_batch: dataset[i + 1],
                              cs1_lc: cstates[0][0], cs1_uc: cstates[0][1],
                              cs2_lc: cstates[1][0], cs2_uc: cstates[1][1],
                              cs3_lc: cstates[2][0], cs3_uc: cstates[2][1],
                              cs4_lc: cstates[3][0], cs4_uc: cstates[3][1], }
                _, cstates, cost, classifications = sess.run([r_train, cellstates, r_cost, r_classification], feed_dict=dictionary)
                accuracy_list.append(np.count_nonzero(np.subtract(np.sign(classifications), np.sign(dataset[i + 1])), arr=0)/20)
                print(f"Epoch: {i} - Cost {cost:3.2f} - Accuracy Sign Coefficient {accuracy_list[-1]*100:3.2f}%")
trainer()