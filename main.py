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


def regressor_procedure(ml_model, comp_data, model_desc):
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
    print_vals(model_desc, train_rmsle, test_rmsle)
    return ml_model


def submit_function(ml_model, comp_data, params, filename):
    if params.submit:
        subm_data = comp_data.submit_dataset() if comp_data.submit_data is None else comp_data.submit_data
        sumbit_log_estimates = ml_model.predict(subm_data)
        sumbit_estimates = invLogTransf(desnorm_y(comp_data, sumbit_log_estimates))
        sumbit = pd.DataFrame({"id": comp_data.raw_submit['id'].values,
                               "trip_duration": sumbit_estimates})
        sumbit.to_csv("output/{}".format(filename), index=False)


def neural_nets_procedure(comp_data, params):
    model = 'MLP neural net'
    # (50,50,50,50,50)
    ml_model = regressor_procedure(
        ml_model=MLPRegressor(hidden_layer_sizes=(100,100,100,100,100,100,100,100,100,100)),
        comp_data=comp_data,
        model_desc=model
    )
    submit_function(ml_model, comp_data, params, 'mlp_submit.csv')




def random_forest_procedure(comp_data, params):
    model = 'random_forest'
    ml_model = regressor_procedure(
        ml_model=RandomForestRegressor(n_estimators=75, n_jobs=-1),
        comp_data=comp_data,
        model_desc=model
    )
    submit_function(ml_model, comp_data, params, 'rf_submit.csv')


def boosted_trees_procedure(comp_data, params):
    model = 'boosted_trees'
    ml_model = regressor_procedure(
        ml_model=GradientBoostingRegressor(max_depth=10),
        comp_data=comp_data,
        model_desc=model
    )
    submit_function(ml_model, comp_data, params, 'boosted_submit.csv')


def support_vector_machines_procedure(comp_data, params):
    model = 'svm'
    ml_model = regressor_procedure(
        ml_model=SVR(),
        comp_data=comp_data,
        model_desc=model
    )
    submit_function(ml_model, comp_data, params, 'rf_submit.csv')


def main(params):
    models = {
        1: neural_nets_procedure,
        2: random_forest_procedure,
        3: boosted_trees_procedure,
        4: support_vector_machines_procedure
    }
    comp_data = CompetitionData(
        frac=kwargs.frac if kwargs.frac is None else 0.7,
        hotstart=kwargs.hotstart if kwargs.hotstart is None else 1)
    comp_data.set_train_dataset()
    models[params.model](comp_data, params)

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("--model", type="int", help="Select model.")
    parser.add_option("--submit", type="int", help="Create submission file.")
    parser.add_option("--hotstart", type="int", help="Use pre-processed file.")
    parser.add_option("--frac", type="float", help="Percentage (in decimal) of dataset to use.")
    kwargs, _ = parser.parse_args(args=None, values=None)
    main(kwargs)
