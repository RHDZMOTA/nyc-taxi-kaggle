# -*- coding: utf-8 -*-
"""
Example file
How to use quanta for machine learning.
import os; os.chdir("/media/rhdzmota/Data/Files/github_mxquants/quanta")
@author: Rodrigo Hernández-Mota
Contact info: rhdzmota@mxquants.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataHandler import Dataset
from neuralNets import mlpRegressor


# Create sample data
x_data = pd.DataFrame({"x1": np.random.rand(5000),
                       "x2": np.arange(5000)**2})
y_data = x_data.apply(lambda x: np.power(np.sqrt(x[1]), x[0]) + 5 *
                      np.sin(np.pi * x[0]), 1)

# Create dataset object
dataset = Dataset(input_data=x_data, output_data=y_data, normalize="minmax")


def numberOfWeights(dataset, hidden_layers, batch_ref=0.7):
    """Get the number of parameters to estimate."""
    n_input = np.shape(dataset.train[0])[-1]
    n_output = np.shape(dataset.train[-1])[-1]
    params = np.prod(np.array([n_input]+list(hidden_layers)+[n_output]))
    n_elements = np.shape(dataset.train[0])[0]
    return params, n_elements, batch_ref*n_elements > params


# Multilayer perceptron
_epochs = 2000
_hdl = [10, 10, 10]

nparams, nelements, not_warn = numberOfWeights(dataset, _hdl)
print(not_warn)
if not not_warn:
    print("Warning: not enough data to train model.")

mlp = mlpRegressor(hidden_layers=_hdl)
mlp.train(dataset=dataset, alpha=0.01, epochs=_epochs)
train = mlp.train_results
train = pd.DataFrame(train)
test = pd.DataFrame(mlp.test(dataset=dataset))

# estimate
y_estimate = mlp.freeEval(dataset.norm_input_data.values)
plt.plot(dataset.norm_input_data[0].values,
         [np.asscalar(i) for i in dataset.norm_output_data.values], "b.")
plt.plot(dataset.norm_input_data[0].values, y_estimate, "r.")
plt.show()

# visualize the training performance
plt.plot(np.arange(_epochs), mlp.epoch_error)
plt.title("MSE per epoch.")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.show()

train.errors.plot(kind="kde")
test.errors.plot(kind="kde")
plt.show()

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
plt.subplot(121)
plt.plot(train.y, train.y, ".g", label="Real vals")
plt.plot(train.y, train.estimates, ".b", label="Estimates")
plt.title("Train")
plt.xlabel("y-values")
plt.ylabel("estimates")
plt.legend()
plt.subplot(122)
plt.plot(test.y, test.y, ".g", label="Real vals")
plt.plot(test.y, test.estimates, ".r", label="Estimates")
plt.title("Test")
plt.xlabel("y-values")
plt.ylabel("estimates")
plt.legend()
plt.show()

# Test with iterative selection of hidden layers
max_hidden = 10
max_neurons = 10
mse = {"hidden": [], "neurons": [], "mse": []}
hidden_vector = [None]
temp_hidden_vector = [None]
best_per_layer = []
for i in range(1, max_hidden+1):
    print("\n")
    for j in range(1, max_neurons+1):
        temp_hidden_vector[i-1] = j
        nparams, nelements, not_warn = numberOfWeights(dataset,
                                                       temp_hidden_vector)
        if not not_warn:
            print("Not viable anymore.")
            break
        hidden_vector[i-1] = j
        print("Evaluating: ({i}, {j})".format(i=i, j=j), sep="\n")
        mse["hidden"].append(i)
        mse["neurons"].append(j)
        mlp = mlpRegressor(hidden_layers=hidden_vector)
        mlp.train(dataset=dataset, alpha=0.01, epochs=_epochs)
        mse["mse"].append(np.mean(mlp.test(dataset=dataset)["square_error"]))
        # print(mse["mse"][-1])
    if not not_warn:
        break
    temp = pd.DataFrame(mse)
    min_mse_arg = temp.query("hidden == {}".format(i)).mse.argmin()
    temp_hidden_vector[i-1] = temp["neurons"].iloc[min_mse_arg]
    hidden_vector[i-1] = temp["neurons"].iloc[min_mse_arg]
    best_per_layer.append(temp["mse"].iloc[min_mse_arg])
    hidden_vector.append(None)
    temp_hidden_vector.append(None)

hidden_vector

plt.plot(np.arange(len(best_per_layer))+1, best_per_layer)
plt.show()


mse_df = pd.DataFrame(mse)
mse_df
x = mse["hidden"]
y = mse["neurons"]
z = mse["mse"]
min_z, max_z = min(z), max(z)
z = [(i-min_z)/(max_z-min_z) for i in z]
plt.scatter(x, y, c=z, s=100)
# plt.gray()
plt.xlabel("Number of hidden layers")
plt.ylabel("Number of neurons at last hl")
plt.grid()
plt.show()


plt.plot(x, mse["mse"])
# plt.gray()
plt.xlabel("Number of hidden layers")
plt.ylabel("mse")
plt.grid()
plt.show()

plt.plot(y, mse["mse"], '.b')
# plt.gray()
plt.xlabel("Number of neurons")
plt.ylabel("mse")
plt.grid()
plt.show()
