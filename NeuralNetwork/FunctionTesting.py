from baseFunctions import *
import MNIST2 as mnist
import NN_layers as layer
import pickle
import numpy as np

training_data = mnist.train_images()
training_labels = mnist.train_labels()
training_labels[np.where(training_labels == 0)] -=1
testing_data = mnist.test_images()
testing_labels = mnist.test_labels()
testing_labels[np.where(testing_labels == 0)] -=1

batch_size = 5000


def model(x):
    x = tf.reshape(x, [-1, 784])
    return geebs_tanh(layer.fcl(x, [784, 100, 100, 10], "FCL1"))


def trainer():
    data_batch = tf.placeholder('float', [batch_size, 28, 28, 1])
    labels_batch = tf.placeholder('float', [batch_size, 10])
    with tf.variable_scope("TEST", reuse=tf.AUTO_REUSE):
        network_classifications = model(data_batch)
    cost_function = tf.math.reduce_mean(tf.math.reduce_sum(tf.square(tf.subtract(network_classifications, labels_batch)), axis=1))
    backprop = tf.train.AdamOptimizer().minimize(cost_function)
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        epoch = 0
        accuracy_list = pickle.load(open(r"C:\Users\Coding Projects\Desktop\Programming Projects\TradingBot\Data\Data.pickle", "rb"))

        experiment = "FCL-gSig-Cost-1,1"
        accuracy_list[experiment].append([])
        data = np.zeros([batch_size, 28, 28, 1])
        labels = np.zeros([batch_size, 10])
        while epoch <= 2000:
            # #Training Cycle
            random = np.random.random_integers(0, 59999, batch_size)
            data[::] = training_data[random, ::]
            labels[::] = training_labels[random, ::]
            _, cost, classification = sess.run([backprop, cost_function, network_classifications], feed_dict={data_batch: data, labels_batch: labels})

            if epoch % 10 == 0:
                data[::] = training_data[:batch_size, ::]
                labels[::] = training_labels[:batch_size, ::]
                cost, classification = sess.run([cost_function, network_classifications], feed_dict={data_batch: data, labels_batch: labels})
                correct = 0
                for i in range(batch_size):
                    if np.argmax(classification, axis=1)[i] == np.argmax(labels, axis=1)[i]:
                        correct += 1
                accuracy = (correct/batch_size)*100
                accuracy_list[experiment][-1].append(accuracy)
                print(f"Epoch {epoch} - Accuracy {accuracy:3.2f}% - Cost {cost:3.2f}")
            epoch += 1
        pickle.dump(accuracy_list, open(r"C:\Users\Coding Projects\Desktop\Programming Projects\TradingBot\Data\Data.pickle", "wb"))


trainer()
