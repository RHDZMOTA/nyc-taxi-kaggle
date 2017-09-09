from optparse import OptionParser
import numpy as np
import pandas as pd

from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR

from utils.performance_functions import rmsle
from utils.data_handler import CompetitionData, desnormalize, invLogTransf


def desnorm_y(comp_data, y_norm):
    return np.array([desnormalize(np.asscalar(i),
                                             comp_data.train_dataset,
                                             comp_data.normalize_method) for i in y_norm])


def drop_one_dim(y):
    first, second = y.shape
    return y.reshape((first, ))


def print_vals(model, train_rmsle, test_rmsle):
    string = 'Model: {}\nTrain: {}\nTest: {}'.format(
        model, train_rmsle, test_rmsle)
    print(string)

def neural_nets_procedure(comp_data, params):
    model = 'MLP neural net'
    # Declare models
    ml_model = MLPRegressor(hidden_layer_sizes=(50,50,50,50,50))
    # Datasets
    x_train, y_train = comp_data.train_dataset.train
    x_test, y_test = comp_data.train_dataset.test
    # Fit
    ml_model.fit(x_train, drop_one_dim(y_train))
    # Estimates
    train_estimates = ml_model.predict(x_train)
    test_estimates = ml_model.predict(x_test)
    # Desnorm
    desnorm_train_y = desnorm_y(comp_data, y_train)
    desnorm_train_yest = desnorm_y(comp_data, train_estimates)
    desnorm_test_y = desnorm_y(comp_data, y_test)
    desnorm_test_yest = desnorm_y(comp_data, test_estimates)
    # RMSLE
    train_rmsle = rmsle(invLogTransf(desnorm_train_yest), invLogTransf(desnorm_train_y))
    test_rmsle = rmsle(invLogTransf(desnorm_test_yest), invLogTransf(desnorm_test_y))
    print_vals(model, train_rmsle, test_rmsle)
    if params.submit:
        subm_data = comp_data.submit_dataset() if comp_data.submit_data is None else comp_data.submit_data
        sumbit_log_estimates = ml_model.predict(subm_data)
        sumbit_estimates = invLogTransf(desnorm_y(comp_data, sumbit_log_estimates))
        sumbit = pd.DataFrame({"id": comp_data.raw_submit['id'].values,
                               "trip_duration": sumbit_estimates})
        sumbit.to_csv("output/mlp_submit.csv", index=False)


def random_forest_procedure(comp_data, params):
    model = 'random_forest'
    # Declare models
    ml_model = RandomForestRegressor()
    # Datasets
    x_train, y_train = comp_data.train_dataset.train
    x_test, y_test = comp_data.train_dataset.test
    # Fit
    ml_model.fit(x_train, drop_one_dim(y_train))
    # Estimates
    train_estimates = ml_model.predict(x_train)
    test_estimates = ml_model.predict(x_test)
    # Desnorm
    desnorm_train_y = desnorm_y(comp_data, y_train)
    desnorm_train_yest = desnorm_y(comp_data, train_estimates)
    desnorm_test_y = desnorm_y(comp_data, y_test)
    desnorm_test_yest = desnorm_y(comp_data, test_estimates)
    # RMSLE
    train_rmsle = rmsle(invLogTransf(desnorm_train_yest), invLogTransf(desnorm_train_y))
    test_rmsle = rmsle(invLogTransf(desnorm_test_yest), invLogTransf(desnorm_test_y))
    print_vals(model, train_rmsle, test_rmsle)
    if params.submit:
        subm_data = comp_data.submit_dataset() if comp_data.submit_data is None else comp_data.submit_data
        sumbit_log_estimates = ml_model.predict(subm_data)
        sumbit_estimates = invLogTransf(desnorm_y(comp_data, sumbit_log_estimates))
        sumbit = pd.DataFrame({"id": comp_data.raw_submit['id'].values,
                               "trip_duration": sumbit_estimates})
        sumbit.to_csv("output/rf_submit.csv", index=False)

def boosted_trees_procedure(comp_data, params):
    model = 'boosted_trees'
    # Declare models
    ml_model = GradientBoostingRegressor()
    # Datasets
    x_train, y_train = comp_data.train_dataset.train
    x_test, y_test = comp_data.train_dataset.test
    # Fit
    ml_model.fit(x_train, drop_one_dim(y_train))
    # Estimates
    train_estimates = ml_model.predict(x_train)
    test_estimates = ml_model.predict(x_test)
    # Desnorm
    desnorm_train_y = desnorm_y(comp_data, y_train)
    desnorm_train_yest = desnorm_y(comp_data, train_estimates)
    desnorm_test_y = desnorm_y(comp_data, y_test)
    desnorm_test_yest = desnorm_y(comp_data, test_estimates)
    # RMSLE
    train_rmsle = rmsle(invLogTransf(desnorm_train_yest), invLogTransf(desnorm_train_y))
    test_rmsle = rmsle(invLogTransf(desnorm_test_yest), invLogTransf(desnorm_test_y))
    print_vals(model, train_rmsle, test_rmsle)
    if params.submit:
        subm_data = comp_data.submit_dataset() if comp_data.submit_data is None else comp_data.submit_data
        sumbit_log_estimates = ml_model.predict(subm_data)
        sumbit_estimates = invLogTransf(desnorm_y(comp_data, sumbit_log_estimates))
        sumbit = pd.DataFrame({"id": comp_data.raw_submit['id'].values,
                               "trip_duration": sumbit_estimates})
        sumbit.to_csv("output/boosted_submit.csv", index=False)


def support_vector_machines_procedure(comp_data, params):
    model = 'svm'
    # Declare models
    ml_model = SVR()
    # Datasets
    x_train, y_train = comp_data.train_dataset.train
    x_test, y_test = comp_data.train_dataset.test
    # Fit
    ml_model.fit(x_train, drop_one_dim(y_train))
    # Estimates
    train_estimates = ml_model.predict(x_train)
    test_estimates = ml_model.predict(x_test)
    # Desnorm
    desnorm_train_y = desnorm_y(comp_data, y_train)
    desnorm_train_yest = desnorm_y(comp_data, train_estimates)
    desnorm_test_y = desnorm_y(comp_data, y_test)
    desnorm_test_yest = desnorm_y(comp_data, test_estimates)
    # RMSLE
    train_rmsle = rmsle(invLogTransf(desnorm_train_yest), invLogTransf(desnorm_train_y))
    test_rmsle = rmsle(invLogTransf(desnorm_test_yest), invLogTransf(desnorm_test_y))
    print_vals(model, train_rmsle, test_rmsle)
    if params.submit:
        subm_data = comp_data.submit_dataset() if comp_data.submit_data is None else comp_data.submit_data
        sumbit_log_estimates = ml_model.predict(subm_data)
        sumbit_estimates = invLogTransf(desnorm_y(comp_data, sumbit_log_estimates))
        sumbit = pd.DataFrame({"id": comp_data.raw_submit['id'].values,
                               "trip_duration": sumbit_estimates})
        sumbit.to_csv("output/rf_submit.csv", index=False)




def main(model, params):
    models = {
        1: neural_nets_procedure,
        2: random_forest_procedure,
        3: boosted_trees_procedure
    }

    comp_data = CompetitionData()
    comp_data.set_train_dataset()

    selected_model = params.model
    while True:
        models[params.model](comp_data, params)
        params.model = int(input('Select another model: '))
        params.submit = int(input('Create submission file? 1/0'))




    pass


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--model", type="int", help="Select model.")
    parser.add_option("--submit", type="int", help="Create submission file.")
    kwargs, _ = parser.parse_args(args=None, values=None)
    main(kwargs.model, kwargs)