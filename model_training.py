#!/usr/bin/env python
import numpy as np
from keras.models import Sequential
from keras.layers import BatchNormalization, Bidirectional, Dropout, Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error


# def run_model_training(x_train, x_valid, x_test, y_train, y_valid, y_test, max_len, dropout_rate, n_layers, nodes,
#                        epochs, batch_size, mode, val_method, val_mode, final_activ='linear'):
