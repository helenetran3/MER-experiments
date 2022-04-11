import os.path
import numpy as np
import csv

from pickle_functions import save_with_pickle, pickle_file_exists, load_from_pickle, save_results_in_csv_file
from dataset_utils import get_tf_dataset

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Dropout, Dense, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix


def build_model(num_features, num_steps, num_layers, num_nodes, dropout_rate, final_activ):
    """
    Build the model described in the paper (cf. README for the reference).
    Works only when the number of steps is the same for all datapoints.

    :param num_features: feature vector size (it should be the same at each step)
    :param num_steps: number of steps in the sequence
    :param num_layers: number of bidirectional layers
    :param num_nodes: number of nodes for the penultimate dense layer
    :param dropout_rate: dropout rate before each dense layer
    :param final_activ: final activation function
    :return: the model built
    """

    model = Sequential()

    if num_layers == 1:
        model.add(BatchNormalization(input_shape=(num_steps, num_features)))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_nodes, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(7, activation=final_activ))

    if num_layers == 2:
        model.add(BatchNormalization(input_shape=(num_steps, num_features)))
        model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(num_steps, num_features))))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_nodes, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(7, activation=final_activ))

    if num_layers == 3:
        model.add(BatchNormalization(input_shape=(num_steps, num_features)))
        model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(num_steps, num_features))))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(num_steps, num_features))))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_nodes, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(7, activation=final_activ))

    return model


def train_model(train_list, valid_list, test_list,
                batch_size, num_epochs, fixed_num_steps, num_layers,
                num_nodes, dropout_rate, final_activ, learning_rate, loss_function,
                val_metric, patience, model_folder, model_name):
    """
    Train the model.

    :param train_list: [x_train, y_train, seg_train] where x_train is a list of arrays of shape (number steps, number
    features), y_train a list arrays of shape (1, 7), and seg_train a list of segment ids (ex: 'zk2jTlAtvSU[1]')
    :param valid_list: [x_valid, y_valid, seg_valid]
    :param test_list: [x_test, y_test, seg_test]
    :param batch_size: Batch size for training
    :param num_epochs: Maximum number of epochs for training
    :param fixed_num_steps: Fixed size for all the sequences (if we keep the original size, this parameter is set to 0)
    :param num_layers: Number of bidirectional layers for the model
    :param num_nodes: Number of nodes for the penultimate dense layer
    :param dropout_rate: Dropout rate before each dense layer
    :param final_activ: Final activation function
    :param learning_rate: Learning rate for training
    :param loss_function: Loss function
    :param val_metric: Metric on validation data to monitor
    :param patience: Number of epochs with no improvement after which the training will be stopped
    :param model_folder: Name of the directory where the models will be saved
    :param model_name: Name of the model to be saved
    :return: history of the model training
    """

    x_train = train_list[0]
    y_train = train_list[1]
    seg_train = train_list[2]
    x_valid = valid_list[0]
    y_valid = valid_list[1]
    seg_valid = valid_list[2]
    num_train_samples = len(y_train)
    num_valid_samples = len(y_valid)
    num_test_samples = len(test_list[1])
    total_data = num_train_samples + num_valid_samples + num_test_samples

    # Create TensorFlow datasets for model training
    with_fixed_length = (fixed_num_steps > 0)
    train_dataset = get_tf_dataset(x_train, y_train, seg_train, batch_size, with_fixed_length, fixed_num_steps,
                                   train_mode=True)
    valid_dataset = get_tf_dataset(x_valid, y_valid, seg_valid, batch_size, with_fixed_length, fixed_num_steps,
                                   train_mode=True)

    # Parameters to save model
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    model_save_name = "{}_l_{}_n_{}_d_{}_b_{}_s_{}.h5".format(model_name, num_layers, num_nodes, dropout_rate,
                                                              batch_size, fixed_num_steps)
    model_save_path = os.path.join(model_folder, model_save_name)

    # Parameters for metric monitoring
    monitor = 'val_loss' if val_metric == 'loss' else 'val_accuracy'
    mode_monitor = 'min' if val_metric == 'loss' else 'max'

    # Initialize callbacks
    checkpoint = ModelCheckpoint(filepath=model_save_path,
                                 verbose=1,
                                 monitor=monitor,
                                 mode=mode_monitor,
                                 save_best_only=True)

    early_stopping = EarlyStopping(monitor=monitor,
                                   patience=patience,
                                   mode=mode_monitor,
                                   restore_best_weights=True)

    # Build model
    num_features = x_train[0].shape[1]
    model = build_model(num_features, fixed_num_steps, num_layers, num_nodes, dropout_rate, final_activ)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer)

    print("\n\n============================== Training Parameters ===========================================")
    print("Number training datapoints: {} ({:.2f}%)".format(num_train_samples, 100 * (num_train_samples / total_data)))
    print("Number validation datapoints: {} ({:.2f}%)".format(num_valid_samples, 100 * (num_valid_samples / total_data)))
    print("Number test datapoints: {} ({:.2f}%)".format(num_test_samples, 100 * (num_test_samples / total_data)))
    print("Batch size:", batch_size)
    print("Number epochs:", num_epochs)
    print("Fixed number of steps:", fixed_num_steps)
    print("Number layers:", num_layers)
    print("Number nodes for the penultimate dense layer:", num_nodes)
    print("Dropout rate:", dropout_rate)
    print("Final activation:", final_activ)
    print("Learning rate:", learning_rate)
    print("Loss function:", loss_function)
    print("Metric to monitor on validation data:", val_metric)
    print("Patience:", patience)

    print("\n\n================================= Model Training =============================================")
    history = model.fit(x=train_dataset,
                        epochs=num_epochs,
                        verbose=1,
                        steps_per_epoch=num_train_samples // batch_size,
                        validation_data=valid_dataset,
                        validation_steps=num_valid_samples // batch_size,
                        callbacks=[checkpoint, early_stopping])

    return history


def evaluate_model(test_list, batch_size, fixed_num_steps, num_layers, num_nodes, dropout_rate, loss_function,
                   model_folder, model_name, csv_folder, csv_name):
    """
    Evaluate the performance of the best model.

    :param test_list: [x_test, y_test, seg_test]
    :param batch_size: Batch size for training
    :param fixed_num_steps: Fixed size for all the sequences (if we keep the original size, this parameter is set to 0)
    :param num_layers: Number of bidirectional layers for the model
    :param num_nodes: Number of nodes for the penultimate dense layer
    :param dropout_rate: Dropout rate before each dense layer
    :param loss_function: Loss function
    :param model_folder: Name of the directory where the best model is saved
    :param model_name: Name of the saved model
    :param csv_folder: Name of the directory where the csv file containing the results is saved
    :param csv_name: Name of the csv file
    """

    x_test = test_list[0]  # each element of shape (29, 409)
    y_test = test_list[1]  # each element of shape (1, 7)
    seg_test = test_list[2]

    # Create TensorFlow test dataset for model evaluation
    with_fixed_length = (fixed_num_steps > 0)
    test_dataset = get_tf_dataset(x_test, y_test, seg_test, batch_size, with_fixed_length, fixed_num_steps,
                                  train_mode=True)

    # Load best model
    parameters_name = "l_{}_n_{}_d_{}_b_{}_s_{}".format(num_layers, num_nodes, dropout_rate,
                                                        batch_size, fixed_num_steps)
    model_save_name = "{}_{}.h5".format(model_name, parameters_name)
    model_save_path = os.path.join(model_folder, model_save_name)
    model = load_model(model_save_path)

    print("\n\n================================= Model Prediction ===========================================")

    true_sc_save_name = "true_scores_{}.h5".format(parameters_name)
    true_cl_save_name = "true_classes_{}.h5".format(parameters_name)
    pred_raw_save_name = "pred_raw_{}.h5".format(parameters_name)
    pred_sc_save_name = "pred_scores_{}.h5".format(parameters_name)
    pred_cl_save_name = "pred_classes_{}.h5".format(parameters_name)

    # Presence score derived from true labels
    true_scores = np.array(y_test).flatten()  # shape (32578,) > each 7 emotions correspond to one segment
    save_with_pickle(true_scores, true_sc_save_name, model_folder)

    # Get all existing presence scores with Label Encoder
    le = LabelEncoder()
    le.fit(true_scores)
    classes = le.classes_
    # Should give the following (23 classes):
    # [-3.         -2.6666667  -2.3333333  -2.          -1.6666666  -1.3333334
    #  -1.         -0.6666667  -0.5        -0.33333334  0.          0.16666667
    #  0.33333334  0.5         0.6666667   0.8333333    1.          1.3333334
    #  1.6666666   2.          2.3333333   2.6666667    3.        ]

    # Classes derived from true labels (useful for metrics including confusion matrix)
    true_classes = le.transform(true_scores)  # values from 0 to 22
    save_with_pickle(true_classes, true_cl_save_name, model_folder)

    # Get raw predictions from the model
    if not pickle_file_exists(pred_raw_save_name, model_folder):  # perform prediction (for debugging)
        num_test_samples = len(y_test)
        pred_raw = model.predict(test_dataset, verbose=1, steps=num_test_samples)
        pred_raw = pred_raw.flatten()  # (32578,)
        save_with_pickle(pred_raw, pred_raw_save_name, model_folder)
    else:
        pred_raw = load_from_pickle(pred_raw_save_name, model_folder)

    # Presence score derived from predictions
    pred_scores = [min(classes, key=lambda x:abs(x-pred_raw[i])) for i in range(pred_raw.shape[0])]
    pred_scores = np.array(pred_scores)  # pred_scores: get the closest class for each continuous value
    save_with_pickle(pred_scores, pred_sc_save_name, model_folder)

    # Classes derived from predictions (useful for metrics including confusion matrix)
    pred_classes = le.transform(pred_scores)  # values from 0 to 22
    save_with_pickle(pred_classes, pred_cl_save_name, model_folder)

    # Confusion matrix
    conf_matrix = confusion_matrix(true_classes, pred_classes)
    save_with_pickle(conf_matrix, 'conf_matrix_' + model_save_name, model_folder)

    # Model evaluation and prediction
    print("\n\n================================= Model Evaluation ===========================================")
    loss_function_val = model.evaluate(test_dataset, verbose=1)
    print("{}: {}".format(loss_function, loss_function_val))

    save_results_in_csv_file(csv_name, csv_folder, num_layers, num_nodes, dropout_rate, batch_size,
                             fixed_num_steps, loss_function, loss_function_val)


