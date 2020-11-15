from baseFunctions import *


def fcl(x, structure, name, keep_prob=1.):
    '''
    :param x: The input tensor of form (batch, neurons)
    :param structure: The network structure e.g. (150, 100, 100, 50) #This determines the output size as also 50 and input size 150.
    :param name: The name for variables created by this lstm model e.g. "FCL 1"
    :return x: The output of the fcl operation.
    '''
    variables = {
        "weights": [],
        "biases": []
    }
    for i in range(1, len(structure)):
        x = tf.nn.dropout(x, keep_prob)
        variables["weights"].append(tf.get_variable(f"{name}.{i+1}.w", (structure[i - 1], structure[i]), initializer=tf.random_normal_initializer()))
        variables["biases"].append(tf.get_variable(f"{name}.{i+1}.b", structure[i], initializer=tf.zeros_initializer))
        x = tf.add(tf.matmul(x, variables["weights"][-1]), variables["biases"][-1])
        if i != len(structure)-1:
            x = tf.nn.sigmoid(x)
    return x


def dense_lstm(x, structure, name, cell_state):
    '''
    :param x: The input tensor of form (batch, neurons)
    :param structure: The addition network structure e.g. (100, 100, 50) #This determines the output size as also 50.
    :param name: The name for variables created by this lstm model e.g. "LSTM 1"
    :param cell_state: This is a list that contains the lower and upper cell states for this lstm. The lower and upper states must each be size (batch, structure[-1])
    :return x: The output of the lstm operation.
    '''
    cell_state_size = int(structure[-1])
    lower_cell, upper_cell = cell_state
    structure = [int(cell_state_size + x.shape[1]), ] + structure

    x = tf.concat([lower_cell, x], 1)
    # #Forget Network
    FN = tf.nn.sigmoid(fcl(x, structure, name+"1-FCL."))
    # #Block Addition Network
    BAN = tf.nn.sigmoid(fcl(x, structure, name+"2-FCL."))
    # #Addition Network
    AN = tf.nn.tanh(fcl(x, structure, name+"3-FCL."))
    # #Filter Output Network
    FON = tf.nn.sigmoid(fcl(x, structure, name+"4-FCL."))
    # #Forget upper cell elements
    upper_cell = tf.multiply(upper_cell, FN)
    # #Addition
    upper_cell = upper_cell + tf.multiply(AN, BAN)
    # #Output
    lower_cell = tf.multiply(tf.nn.tanh(upper_cell), FON)
    return lower_cell, upper_cell


def norm(x):
    '''
    :param x: Tensor to be normalised (batch_size, y, x, z)
    :return: Normalised tensor
    '''
    return tf.divide(tf.subtract(x, tf.math.reduce_mean(x, axis=0)), tf.math.reduce_std(x, axis=0))


def leaky_conv(x, weights, biases, stride):
    '''
    :param x: Input to be convoluted (batch_size, y, x, z)
    :param weights: Weights to use on convoluted image (y, x, z, num)
    :param biases: Bias to add to convolved image (num)
    :param stride: List to indicate the stride length (y, x)
    :return: Convolved image
    '''
    if x.shape[0] >= 20:
        return tf.nn.leaky_relu(norm(tf.add(tf.nn.conv2d(x, weights, strides=[1, stride[0], stride[1], 1], padding="SAME"), biases)))
    else:
        return tf.nn.leaky_relu(tf.add(tf.nn.conv2d(x, weights, strides=[1, stride[0], stride[1], 1], padding="SAME"), biases))


def leaky_deconv(x, weights, biases, stride):
    if x.shape[0] >= 20:
        return tf.nn.leaky_relu(tf.nn.conv2d_transpose(tf.add(norm(x), biases), weights, output_shape=(x.shape[0], x.shape[1]*stride[0], x.shape[2]*stride[1], weights.shape[2]), strides=[1, stride[0], stride[0], 1], padding="SAME"))
    else:
        return tf.nn.leaky_relu(tf.nn.conv2d_transpose(tf.add(x, biases), weights, output_shape=(x.shape[0], x.shape[1] * stride[0], x.shape[2] * stride[1], weights.shape[2]), strides=[1, stride[0], stride[0], 1], padding="SAME"))


def leaky_conv2(x, filter_size, stride, name):
    '''
    :param x: Input to be convoluted (batch_size, y, x, z)
    :param filter_size: Weights structure to use on convoluted image (y, x, in, out)
    :param stride: List to indicate the stride length (y, x)
    :param name: The name for variables created by this lstm model e.g. "LSTM 1"
    :return: Convolved image
    '''
    weights = tf.get_variable(f"{name}.w", filter_size, initializer=tf.random_normal_initializer)
    biases = tf.get_variable(f"{name}.b", filter_size[3], initializer=tf.zeros_initializer)
    return tf.nn.leaky_relu(tf.add(tf.nn.conv2d(x, weights, strides=[1, stride[0], stride[1], 1], padding="SAME"), biases))


def modified_dense_lstm(x, structure, name, cell_state):
    '''
    :param x: The input tensor of form (batch, neurons)
    :param structure: The addition network structure e.g. (100, 100, 50) #This determines the output size as also 50.
    :param name: The name for variables created by this lstm model e.g. "LSTM 1"
    :param cell_state: This is a list that contains the lower and upper cell states for this lstm. The lower and upper states must each be size (batch, structure[-1])
    :return x: The output of the lstm operation.
    '''
    cell_state_size = int(structure[-1])
    lower_cell, upper_cell = cell_state
    structure = [int(cell_state_size + x.shape[1]), ] + structure

    x = tf.concat([lower_cell, x], 1)
    # #Forget Network
    FN = tf.nn.sigmoid(fcl(x, structure, name+"1-FCL."))
    # #Block Addition Network
    BAN = tf.nn.sigmoid(fcl(x, structure, name+"2-FCL."))
    # #Addition Network
    AN = tf.nn.tanh(fcl(x, structure, name+"3-FCL."))
    # #Filter Output Network
    FON = tf.nn.sigmoid(fcl(x, structure, name+"4-FCL."))
    # #Forget upper cell elements
    upper_cell = tf.multiply(upper_cell, FN)
    # #Addition
    upper_cell = upper_cell + tf.multiply(AN, BAN)
    # #Output
    lower_cell = tf.multiply(tf.nn.tanh(upper_cell), FON)
    return lower_cell, upper_cell