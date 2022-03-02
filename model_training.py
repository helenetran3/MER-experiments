#!/usr/bin/env python
################################################################################
#       The Edinburgh G25 Multimodal DNN Emotion Recognition System
#                https://github.com/rhoposit/emotionChallenge
#
#                Centre for Speech Technology Research
#                     University of Edinburgh, UK
#                      Copyright (c) 2017-2018
#                        All Rights Reserved.
#
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute
#  this software and its documentation without restriction, including
#  without limitation the rights to use, copy, modify, merge, publish,
#  distribute, sublicense, and/or sell copies of this work, and to
#  permit persons to whom this work is furnished to do so, subject to
#  the following conditions:
#
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   - The authors' names may not be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
#  THIS SOFTWARE.
################################################################################

import numpy as np
from keras.models import Sequential
from keras.layers import BatchNormalization, Bidirectional, Dropout, Dense, LSTM
from keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error


def get_class_MAE(truth, preds):
    ref = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise']
    class_MAE = []
    truth = np.array(truth)
    preds = np.array(preds)
    for i in range(len(truth[0])):
        T = truth[:, i]
        P = preds[:, i]
        class_MAE.append(mean_absolute_error(T, P))
    outstring = ""
    for i in range(len(class_MAE)):
        o = ref[i] + "=" + str(class_MAE[i]) + "\n"
        outstring += o
    return outstring


def run_model_training(x_train, x_valid, x_test, y_train, y_valid, y_test, max_len, dropout_rate, n_layers, nodes,
                       epochs, batch_size, mode, val_method, val_mode, final_activ='linear'):
    outfile = "final_sweep/blstm_" + mode + "_" + str(n_layers) + "_" + str(max_len) + "_" + str(dropout_rate)
    experiment_prefix = "blstm_early_fusion"
    logs_path = "regression_logs/"
    experiment_name = "{}_n_{}_dr_{}_nl_{}_ml_{}".format(experiment_prefix, nodes, dropout_rate, n_layers, max_len)

    model = Sequential()

    if n_layers == 1:
        model.add(BatchNormalization(input_shape=(max_len, x_train.shape[2])))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(nodes, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(6, activation=final_activ))

    if n_layers == 2:
        model.add(BatchNormalization(input_shape=(max_len, x_train.shape[2])))
        model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(max_len, x_train.shape[2]))))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(nodes, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(6, activation=final_activ))

    if n_layers == 3:
        model.add(BatchNormalization(input_shape=(max_len, x_train.shape[2])))
        model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(max_len, x_train.shape[2]))))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(max_len, x_train.shape[2]))))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(nodes, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(6, activation=final_activ))

    model.compile('adam', loss='mean_absolute_error')

    early_stopping = EarlyStopping(monitor=val_method,
                                   min_delta=0,
                                   patience=10,
                                   verbose=1, mode=val_mode)
    callbacks_list = [early_stopping]
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=[x_valid, y_valid],
              callbacks=callbacks_list)

    out = open(outfile, "wb")
    out.write("---------ORIGINAL predictions (not scaled or bounded)---------")
    preds = model.predict(x_test)
    mae = mean_absolute_error(y_test, preds)
    class_mae = get_class_MAE(y_test, preds)
    out.write("Test Sklearn MAE: " + str(mae) + "\n")
    out.write("Per-class MAE: " + str(class_mae) + "\n")
    out.write("dropout_rate=" + str(dropout_rate) + "\n")
    out.write("n_layers=" + str(n_layers) + "\n")
    out.write("max_len=" + str(max_len) + "\n")
    out.write("nodes=" + str(nodes) + "\n")
    out.write("mode=" + str(mode) + "\n")
    # out.write("num_train=" + str(len(train_set_ids)) + "\n")
    # out.write("num_valid=" + str(len(valid_set_ids)) + "\n")
    # out.write("num_test=" + str(len(test_set_ids)) + "\n")
    out.close()
