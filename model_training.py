import os.path
import numpy as np

from pickle_functions import save_with_pickle, pickle_file_exists, load_from_pickle
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
                   model_folder, model_name):
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

    # Model prediction
    print("\n\n================================= Model Prediction ===========================================")

    pred_save_name = "predictions_{}.h5".format(parameters_name)
    true_save_name = "true_labels_{}.h5".format(parameters_name)

    ## True labels
    true_labels = np.array(y_test).flatten()
    save_with_pickle(true_labels, true_save_name, model_folder)
    # print("np.array(y_test).shape", np.array(y_test).shape)
    # print("true_labels.shape", true_labels.shape)
    # print("true_labels:", true_labels[:100])

    ## Get all presence scores with Label Encoder
    le = LabelEncoder()
    le.fit(true_labels)
    classes = le.classes_ # [-3.         -2.6666667  -2.3333333  -2.         -1.6666666  -1.3333334
                            # -1.         -0.6666667  -0.5        -0.33333334  0.          0.16666667
                            # 0.33333334  0.5         0.6666667   0.8333333   1.          1.3333334
                            # 1.6666666   2.          2.3333333   2.6666667   3.        ] (23 classes)
    # print("classes", classes)
    true_labels_classes = le.transform(true_labels)  # values from 0 to 22
    # print("true_labels_classes:", true_labels_classes[:100])

    ## Predictions
    if not pickle_file_exists(pred_save_name, model_folder):  # perform prediction (for debugging)
        num_test_samples = len(y_test)
        predictions = model.predict(test_dataset, verbose=1, steps=num_test_samples)
        predictions = predictions.flatten()
        save_with_pickle(predictions, pred_save_name, model_folder)
    else:
        predictions = load_from_pickle(pred_save_name, model_folder)

    # print("predictions.shape", predictions.shape)  # (32578,)
    # print("predictions:", predictions[:100])
    pred_labels = [min(classes, key=lambda x:abs(x-predictions[i])) for i in range(predictions.shape[0])]
    pred_labels = np.array(pred_labels)  # pred_labels: get the closest class for each continuous value
    # print("predictions_labels shape:", predictions_labels.shape)
    # print("predictions_labels:", predictions_labels[:100])
    pred_labels_classes = le.transform(pred_labels)  # values from 0 to 22
    # print("pred_labels_classes:", pred_labels_classes[:100])

    ## Confusion matrix
    conf_matrix = confusion_matrix(true_labels_classes, pred_labels_classes)
    save_with_pickle(conf_matrix, 'conf_matrix' + model_save_name, model_folder)
    # print(conf_matrix)

    # Model evaluation and prediction
    print("\n\n================================= Model Evaluation ===========================================")
    print("{}: {}".format(loss_function, model.evaluate(test_dataset, verbose=1)))
