from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import random


# Parameters
learning_rate = 0.0001
training_epochs = 40
batch_size = 10
display_step = 1


def lrelu(x, alpha=0.01):
  return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


class Autoencoder():

    def __init__(self, layers=(100, 20, 10)):
        n_input, n_hidden_1, n_hidden_2 = layers
        self._weights = {
            'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
            'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
        }
        self._biases = {
            'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
            'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([n_input])),
        }

    def _encoder(self, x):
        # Encoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self._weights['encoder_h1']),
                                       self._biases['encoder_b1']))
        # Encoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self._weights['encoder_h2']),
                                       self._biases['encoder_b2']))
        return layer_2

    def _decoder(self, x):
        # Decoder Hidden layer with sigmoid activation #1
        layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, self._weights['decoder_h1']),
                                       self._biases['decoder_b1']))
        # Decoder Hidden layer with sigmoid activation #2
        layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, self._weights['decoder_h2']),
                                       self._biases['decoder_b2']))
        return layer_2

    def fit_and_predict(self, X, Y):
        input = tf.placeholder("float", (None, X.shape[1]))

        # Construct model
        encoder_op = self._encoder(input)
        decoder_op = self._decoder(encoder_op)

        # Prediction
        y_pred = decoder_op
        # Targets (Labels) are the input data.
        y_true = input

        # Define loss and optimizer, minimize the squared error
        cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        # Initializing the variables
        init = tf.global_variables_initializer()

        # Launch the graph
        previous_cost = 1e18
        with tf.Session() as sess:
            sess.run(init)
            # Training cycle
            for epoch in range(training_epochs):
                # Run optimization op (backprop) and cost op (to get loss value)
                np.random.shuffle(X)
                _, c = sess.run([optimizer, cost], feed_dict={input: X})
                # Display logs per epoch step
                if epoch % display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1),
                          "cost=", "{:.9f}".format(c))
#                 if np.abs(previous_cost - c) < 0.01:
#                     print("Not improving, stopping.")
#                     break
#                 previous_cost = c

            print("Optimization Finished!")

            # Applying encode and decode over test set
            encode_decode = sess.run(y_pred, {input: Y})
        return encode_decode


def main():
    n_samples, n_features = 1000, 2200
    n_tests = 30

    autoencoder = Autoencoder(layers=(n_features, int(n_features / 2), int(n_features / 3)))

    X = np.array(np.ones((n_samples, n_features)), dtype=np.float32)
    Y = np.array(np.ones((n_tests, n_features)), dtype=np.float32)
    prediction = autoencoder.fit_and_predict(X, Y)


if __name__ == "__main__":
    main()
