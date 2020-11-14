import tensorflow as tf


def eye(x):
    return tf.eye(tf.cast(x.shape[0], tf.int32))


def unitise(x):
    return tf.matmul(tf.diag(tf.divide(1.0, tf.sqrt(tf.reduce_sum(tf.square(x), axis=1)))), x)


def matrix_magnitude(x):
    return tf.diag(tf.sqrt(tf.reduce_sum(tf.square(x), axis=1)))


def k_delta(x):
    return tf.math.multiply(eye(x), x)


def geebs_tanh(x):
    return tf.matmul(tf.math.tanh(matrix_magnitude(x)), unitise(x))


def maxim_relu(x, alpha=0.1, beta=10.0):
    return tf.matmul(
            tf.add(
                tf.keras.backend.maximum(
                    tf.multiply(alpha, tf.subtract(matrix_magnitude(x), tf.multiply(beta, eye(x)))),
                    tf.multiply(alpha, matrix_magnitude(x))
                    ),
                tf.multiply(tf.multiply(alpha, beta), eye(x))
                ),
            unitise(x)
        )


def bounded_geebs_tanh(x):
    n = tf.reshape(tf.reduce_max(tf.abs(x), axis=1), [1, tf.reduce_max(tf.abs(x), axis=1).shape[0]])
    n = tf.linalg.inv(tf.multiply(2.0, tf.pow(k_delta(tf.matmul(tf.transpose(n), n)), 0.5)))
    return tf.add(
                tf.matmul(
                    tf.keras.backend.minimum(
                        n,
                        tf.multiply(0.5, k_delta(tf.ones_like(n)))
                    ),
                    x
                ),
        tf.multiply(0.5, tf.ones_like(x)))


# #Modified fcl vs fcl
# #Modified conv vs conv
# #Modified lstm vs lstm
# #Modified lstm 2 vs lstm