import math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import tensorflow as tf


def random_dataset(X, Y, seed=0):
    m = X.shape[1]  # number of training examples
    np.random.seed(seed)
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))
    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case
    return shuffled_X, shuffled_Y


def create_placeholders(n_x, n_y):
    X = tf.placeholder(tf.float32, [n_x, None])
    Y = tf.placeholder(tf.float32, [n_y, None])
    return X, Y


def initialize_parameters(n_x, n_y):
    W1 = tf.get_variable("W1", [5, n_x], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable("b1", [5, 1], initializer=tf.zeros_initializer())
    W2 = tf.get_variable("W2", [3, 5], initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable("b2", [3, 1], initializer=tf.zeros_initializer())
    W3 = tf.get_variable("W3", [n_y, 3], initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.get_variable("b3", [n_y, 1], initializer=tf.zeros_initializer())
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    Z1 = tf.matmul(W1, X) + b1
    A1 = tf.nn.leaky_relu(Z1)
    Z2 = tf.matmul(W2, A1) + b2
    A2 = tf.nn.leaky_relu(Z2)
    Z3 = tf.matmul(W3, A2) + b3
    return Z3


def one_hot_matrix(labels, C):
    # Create a tf.constant equal to C (depth), name it 'C'. (approx. 1 line)

    # Use tf.one_hot, be careful with the axis (approx. 1 line)
    one_hot_matrix = tf.one_hot(labels, C, axis=-1)
    # Create the session (approx. 1 line)
    sess = tf.Session()
    # Run the session (approx. 1 line)
    one_hot = sess.run(one_hot_matrix)
    # Close the session (approx. 1 line). See method 1 above.
    sess.close()
    return one_hot


def compute_cost(Z3, Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return cost


def model(X_train, Y_train, X_test, Y_test, message, learning_rate=0.0001,
          num_epochs=10000, print_cost=True):
    ops.reset_default_graph()  # to be able to return the model without overwriting tf variables
    (n_x, m) = X_train.shape  # (n_x: input size, m : number of examples in the train set)
    n_y = Y_train.shape[0]  # n_y : output size
    costs = []  # To keep track of the cost
    X, Y = create_placeholders(n_x, n_y)

    # Initialize parameters
    parameters = initialize_parameters(n_x, n_y)
    # Forward propagation: Build the forward propagation in the tensorflow graph
    Z3 = forward_propagation(X, parameters)

    # Cost function: Add cost function to tensorflow graph
    cost = compute_cost(Z3, Y)

    # Backpropagation: Define the tensorflow optimizer. Use an AdamOptimizer.
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1).minimize(cost)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=1e-02, ).minimize(cost)

    # Initialize all the variables
    init = tf.global_variables_initializer()

    # Start the session to compute the tensorflow graph
    with tf.Session() as sess:
        seed = 3
        # Run the initialization
        sess.run(init)
        # Do the training loop
        for epoch in range(num_epochs):
            epoch_cost = 0.  # Defines a cost related to an epoch
            seed = seed + 1
            shuffled_X, shuffled_Y = random_dataset(X_train, Y_train)
            _, epoch_cost = sess.run([optimizer, cost], feed_dict={X: shuffled_X, Y: shuffled_Y})
            if print_cost:
                if epoch % 1000 == 0:
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if epoch % 10 == 0:
                    costs.append(epoch_cost)

        # plot the cost
        if print_cost:
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per tens)')
            # plt.title("Learning rate =" + str(learning_rate))
            plt.title(message)
            plt.show()

        # lets save the parameters in a variable
        parameters = sess.run(parameters)
        # print("Parameters have been trained!")

        # Calculate the correct predictions
        correct_prediction = tf.equal(tf.argmax(Z3), tf.argmax(Y))

        # Calculate accuracy on the test set
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        accurate = accuracy.eval({X: X_test, Y: Y_test})
        print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print("Test Accuracy:", accurate)

        return parameters, accurate
