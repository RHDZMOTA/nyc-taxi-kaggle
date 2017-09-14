# NYC Taxi Trip Duration

[Kaggle competence](https://www.kaggle.com/c/nyc-taxi-trip-duration) consisting of determining the duration of a trip
in nyc given a set of related variables (thus, a regression problem). 

In this repo I explore different solutions. You can see the leaderboard and some 
kernels [here](https://www.kaggle.com/c/nyc-taxi-trip-duration/leaderboard).


Lead developer: [rhdzmota](rhdzmota@mxquants.com)

## Available models

This repo aims to provide a simple CLI interface to train different ML-Models for
regression. The default available models are:

1. Multi-layer neural net perceptron.
2. Random Forest.
3. Boosted Trees.
4. Support Vector Machines.

## Usage

1. Clone this repo as following to ensure submodules availability:

* ```git clone --recursive https://github.com/rhdzmota/nyc-taxi-trip-duration.git``` 

2. In a terminal run:

* ```python main.py --model i```
* Replace i with the id of the model (see **available models**).

## Advanced usage

There are several hyper-parameters available. The parameter will be used if the
depending on the selected model. 

* `--submit 1` : generates the submission csv.
* `--hidden (10, 10, 10)` : hidden layers in a MLP.
* `--epochs 500` : number of training epochs.

## Setup

Run the following in the root project:
* ```virtualenv venv```
* ```source activate venv```
* ```pip install -r requirements.txt```
* ```python setup.py```


## TODO

* Usage of virtualenv.
* etc