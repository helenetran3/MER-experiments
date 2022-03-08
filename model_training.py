#!/usr/bin/env python
import numpy as np
from keras.models import Sequential
from keras.layers import BatchNormalization, Bidirectional, Dropout, Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error


def build_model(x_train, n_layers, n_steps, n_nodes, dropout_rate, final_activ="linear"):
    """
    Build the model described in the paper (cf. README).

    :param x_train: training data
    :param n_layers: number of layers
    :param n_steps: number of steps in the sequence
    :param n_nodes: number of nodes for the penultimate dense layer
    :param dropout_rate: dropout rate before each dense layer
    :param final_activ: final activation function
    :return: the model built
    """

    model = Sequential()

    if n_layers == 1:
        model.add(BatchNormalization(input_shape=(n_steps, x_train.shape[2])))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(n_nodes, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(6, activation=final_activ))

    if n_layers == 2:
        model.add(BatchNormalization(input_shape=(n_steps, x_train.shape[2])))
        model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(n_steps, x_train.shape[2]))))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(n_nodes, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(6, activation=final_activ))

    if n_layers == 3:
        model.add(BatchNormalization(input_shape=(n_steps, x_train.shape[2])))
        model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(n_steps, x_train.shape[2]))))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(n_steps, x_train.shape[2]))))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(n_nodes, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(6, activation=final_activ))

    return model


# def run_model_training(x_train, x_valid, x_test, y_train, y_valid, y_test, n_steps, dropout_rate, n_layers, n_nodes,
#                        epochs, batch_size, mode, val_method, val_mode, final_activ='linear'):
