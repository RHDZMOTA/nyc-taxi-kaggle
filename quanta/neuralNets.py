#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Neural Networks.

Simple neural networks using tensorflow
@author: Rodrigo Hernández-Mota
rhdzmota@mxquants.com
"""


import numpy as np
import tensorflow as tf


class mlpBase(object):
    """Base guideline for MLPs."""

    def __init__(self, hidden_layers, activation_function=[]):
        """Initialize mlp Regressor."""
        # arguments to lists
        hidden_layers_list = [hidden_layers] if type(hidden_layers) == str \
            else list(hidden_layers)
        activ_funcs_list = [activation_function] if type(activation_function) \
            == str else list(activation_function)

        self.hidden_layers = hidden_layers_list
        if len(hidden_layers_list) == len(activ_funcs_list):
            self.activation = [self._activation(i)
                               for i in activation_function] + \
                                            self._lastActivation(self._type)
        elif len(hidden_layers_list) > len(activ_funcs_list):
            new_activ_func = []
            for i in range(len(hidden_layers_list)):
                new_activ_func.append("relu")
            self.activation = new_activ_func + self._lastActivation(self._type)
        else:
            print("Legnth mismatch among hidden_layers and activation_" +
                  "function.")

    def _lastActivation(self, _type):
        return [""] if "reg" in _type else ["sigmoid"]

    def _activation(self, _string):
        functions = {"sigmoid": tf.nn.sigmoid,
                     "relu": tf.nn.relu,
                     "softmax": tf.nn.softmax,
                     "": lambda x: x}
        return functions.get(_string)

    def _initilizeModel(self, n_inputs, n_outputs=1):
        self.weights, self.bias = {}, {}
        _c = 0
        last_size = n_inputs
        # no hidden layers
        if not len(self.hidden_layers):
            self.weights[_c] = tf.Variable(tf.zeros([last_size, n_outputs]))
            self.bias[_c] = tf.Variable(tf.zeros([n_outputs]))
        # at least 1 hidden layer
        else:
            for i in self.hidden_layers+[n_outputs]:
                self.weights[_c] = tf.Variable(tf.zeros([last_size, i]))
                self.bias[_c] = tf.Variable(tf.zeros([i]))
                last_size = i  # change "columns" to "rows" for next layer.
                _c += 1  # increase position
        return self.weights, self.bias

    def prediction(self, x, weights, biases):
        """Prediction function."""
        layers = [x]
        for i, z in zip(sorted(weights.keys()), self.activation):
            phi = self._activation(z)
            layers.append(phi(tf.add(tf.matmul(
                                                layers[-1],
                                                weights[i]),
                                     biases[i])))
        self.layers = layers
        return layers[-1]


class mlpRegressor(mlpBase):
    """MultiLayer Perceptron Regressor using tensorflow."""

    # Variables
    _type = "regressor"

    def cost(self, pred, y):
        """Cost function."""
        return tf.reduce_mean(tf.squared_difference(pred, y))

    def train(self, dataset, alpha=0.01, epochs=500):
        """Train the model."""
        self.n_inputs = np.shape(dataset.input_data)[-1]
        self.n_outputs = np.shape(dataset.output_data)[-1]
        y = tf.placeholder(tf.float32, [None, self.n_outputs])
        x = tf.placeholder(tf.float32, [None, self.n_inputs])

        tf_vars = {}
        self.weights, self.biases = self._initilizeModel(self.n_inputs,
                                                         self.n_outputs)
        tf_vars["pred"] = self.prediction(x, self.weights, self.biases)
        tf_vars["cost"] = self.cost(tf_vars["pred"], y)
        tf_vars["opt"] = tf.train.AdamOptimizer(alpha).minimize(
                                                    tf_vars["cost"])
        tf_vars["mse"] = tf.squared_difference(tf_vars["pred"], y)
        self.epoch_error = []
        self.init_op = tf.global_variables_initializer()
        result = {"y": [np.asscalar(i) for i in dataset.train[-1]]}
        with tf.Session() as sess:
            sess.run(self.init_op)
            for epoch in range(epochs):
                x_batch, y_batch = dataset.nextBatch()
                sess.run(tf_vars["opt"], feed_dict={x: x_batch,
                                                    y: y_batch})
                self.epoch_error.append(np.mean(sess.run(tf_vars["mse"],
                                        feed_dict={x: x_batch, y: y_batch})))
            vals = sess.run(tf_vars["pred"], feed_dict={x: dataset.train[0]})
            self.np_w = sess.run(self.weights)
            self.np_b = sess.run(self.biases)
            self.tf_vars = tf_vars
        result["estimates"] = [np.asscalar(i) for i in vals]
        result["errors"] = [j-i for i, j in zip(
                                    result["estimates"], result["y"])]
        result["square_error"] = [i**2 for i in result["errors"]]
        self.train_results = result

    def freeEval(self, free_input):
        """Custom input to evaluate."""
        x = tf.placeholder(tf.float32, [None, self.n_inputs])
        weights = self.np_w
        biases = self.np_b
        pred = self.prediction(x, weights, biases)
        self.init_op = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(self.init_op)
            vals = sess.run(pred, feed_dict={x: free_input})
        return vals

    def evaluate(self, dataset):
        """Evaluate train data from a given dataset."""
        x_test = dataset.test[0]
        return self.freeEval(x_test)

    def test(self, dataset):
        """Test data from a dataset object."""
        vals = self.evaluate(dataset)
        real_y = dataset.test[-1]
        result = {"estimates": [np.asscalar(i) for i in vals],
                  "y": [np.asscalar(i) for i in real_y]}
        result["errors"] = [j-i for i, j in zip(
                                    result["estimates"], result["y"])]
        result["square_error"] = [i**2 for i in result["errors"]]
        self.test_results = result
        return result
