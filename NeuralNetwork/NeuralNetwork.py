import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
from baseFunctions import *
from NN_layers import *


def medium_extractor(alpha0, cellstates):
    # #[?, 20]
    cellstate_1, cellstate_2, cellstate_3, cellstate_4 = cellstates

    alpha1 = fcl(alpha0, [20, 64, 64],"a-FCL-1a")
    alpha1, upper_cell1 = dense_lstm(alpha1, [128, 128, 64], "a-LSTM-1b", cellstate_1)

    alpha2 = fcl(alpha1, [64, 128, 128],"a-FCL-2a")
    alpha2, upper_cell2 = dense_lstm(alpha2, [256, 256, 128], "a-LSTM-2b", cellstate_2)

    alpha3 = fcl(alpha2, [128, 256, 256],"a-FCL-3a")
    alpha3, upper_cell3 = dense_lstm(alpha3, [512, 512, 256], "a-LSTM-3b", cellstate_3)

    beta0 = fcl(alpha0, [20, 256, 256, 128], "b-FCL-0a")
    beta1 = fcl(alpha1, [64, 256, 256, 128], "b-FCL-1a")
    beta2 = fcl(alpha2, [128, 256, 256, 128], "b-FCL-2a")
    beta3 = fcl(alpha3, [256, 256, 256, 128], "b-FCL-3a")

    attention = fcl(tf.concat([beta0, beta1, beta2, beta3], axis=1), [512, 256, 4], "Attention")
    a0, a1, a2, a3 = tf.split(attention, num_or_size_splits=4, axis=1)

    alpha4 = fcl(tf.concat([tf.multiply(a0, beta0),
                               tf.multiply(a1, beta1),
                               tf.multiply(a2, beta2),
                               tf.multiply(a3, beta3)], axis=1), [512, 256, 128], "Attention")
    output, upper_cell4 = fcl(dense_lstm(alpha4, [128, 512, 512], "a-LSTM-4b", cellstate_4), [512, 256, 256, 20], "FCL-Final")

    cellstates = [[alpha1, upper_cell1], [alpha2, upper_cell2], [alpha3, upper_cell3], [output, upper_cell4]]

    return output, cellstates, attention

