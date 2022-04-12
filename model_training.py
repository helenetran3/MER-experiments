import os.path
import numpy as np
import csv

from pickle_functions import save_with_pickle, pickle_file_exists, load_from_pickle, save_results_in_csv_file
from dataset_utils import get_tf_dataset

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Dropout, Dense, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam

from sklearn.preprocessing import LabelEncoder, label_binarize, MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score


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


def get_class_from_presence_score(score_vec, with_neutral_class, only_dominant):
    """
    Get the dominant emotion class(es) for each test segment based on the presence scores of emotions.
    Column indexes: 0:happy, 1:sad, 2:anger, 3:surprise, 4:disgust, 5:fear, (6:neutral)

    :param score_vec: array of shape (test_size,7) containing the sentiment + the presence score of 6 emotions
    :param with_neutral_class: whether we add a neutral class or not
    :param only_dominant: only keep dominant classes (else keep any emotion present)
    :return: Binary array of shape (test_size,7) if we add neutral class, else array of shape (test_size,6).
    """

    list_dominant_emotions = []
    test_size = score_vec.shape[0]

    for seg_id in range(test_size):  # we check every segment

        seg_emotions = score_vec[seg_id, 1:]  # we removed the first value which corresponds to sentiment
        max_score = max(seg_emotions)

        if max_score == 0:  # neutral case
            list_dominant_emotions.append([6]) if with_neutral_class else list_dominant_emotions.append([])

        else:  # there is at least one emotion
            if only_dominant:
                # the following takes the index(es) of the maximum value in the presence score vector of emotions
                list_dominant_emotions.append([i for i, val in enumerate(seg_emotions) if val == max_score])
            else:
                list_dominant_emotions.append([i for i, val in enumerate(seg_emotions) if val != 0])

    num_classes = 7 if with_neutral_class else 6
    mlb = MultiLabelBinarizer(classes=list(range(num_classes)))
    bin_array_dominant_emotions = mlb.fit_transform(list_dominant_emotions)

    return bin_array_dominant_emotions


def evaluate_model(test_list, batch_size, fixed_num_steps, num_layers, num_nodes, dropout_rate, loss_function,
                   model_folder, model_name, csv_folder, csv_name, predict_neutral_class, round_decimals):
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
    :param round_decimals: Number of decimals to be rounded for metrics.
    """

    # Load best model
    parameters_name = "l_{}_n_{}_d_{}_b_{}_s_{}".format(num_layers, num_nodes, dropout_rate,
                                                        batch_size, fixed_num_steps)
    model_save_name = "{}_{}.h5".format(model_name, parameters_name)
    model_save_path = os.path.join(model_folder, model_save_name)
    model = load_model(model_save_path)

    # Array names to save
    true_sc_save_name = "true_scores_{}.h5".format(parameters_name)
    true_cl_dom_save_name = "true_classes_dom_{}.h5".format(parameters_name)
    true_cl_all_save_name = "true_classes_all_{}.h5".format(parameters_name)
    pred_raw_save_name = "pred_raw_{}.h5".format(parameters_name)
    pred_sc_save_name = "pred_scores_{}.h5".format(parameters_name)
    pred_cl_dom_save_name = "pred_classes_dom_{}.h5".format(parameters_name)
    pred_cl_all_save_name = "pred_classes_all_{}.h5".format(parameters_name)

    # Extract x, y and seg_ids for test set
    x_test = test_list[0]  # each element of shape (29, 409)
    y_test = test_list[1]  # each element of shape (1, 7)
    seg_test = test_list[2]

    # True presence scores
    true_scores = np.reshape(np.array(y_test), (-1, 7))  # (4654, 7), possible values: 0, 0.33, 0.66, 1
    save_with_pickle(true_scores, true_sc_save_name, model_folder)
    # True classes: binary arrays of shape (4654, 7)
    true_classes_dom = get_class_from_presence_score(true_scores, predict_neutral_class, only_dominant=True)
    true_classes_all = get_class_from_presence_score(true_scores, predict_neutral_class, only_dominant=False)
    save_with_pickle(true_classes_dom, true_cl_dom_save_name, model_folder)
    save_with_pickle(true_classes_all, true_cl_all_save_name, model_folder)
    # print(true_classes_dom[:10, :])
    # print(true_classes_all[:10, :])
    # print(true_scores[:10, 1:])

    # TODO Confusion matrix from that

    # Create TensorFlow test dataset for model evaluation
    with_fixed_length = (fixed_num_steps > 0)
    test_dataset = get_tf_dataset(x_test, y_test, seg_test, batch_size, with_fixed_length, fixed_num_steps,
                                  train_mode=False)

    print("\n\n================================= Model Prediction ===========================================")

    # Get raw score predictions from the model
    if not pickle_file_exists(pred_raw_save_name, model_folder):  # perform prediction (for debugging)
        num_test_samples = len(y_test)
        pred_raw = model.predict(test_dataset, verbose=1, steps=num_test_samples)
        pred_raw = pred_raw.flatten()  # (32578,)
        save_with_pickle(pred_raw, pred_raw_save_name, model_folder)
    else:
        pred_raw = load_from_pickle(pred_raw_save_name, model_folder)

    # pred_raw[1] = 2.1  # For debugging
    # pred_raw[2] = 1.5

    # Get all existing presence scores with Label Encoder
    le = LabelEncoder()
    le.fit(true_scores[:, 1:].flatten())
    classes = le.classes_
    # Presence score derived from predictions (the closest value among [0, 0.33, 0.66, 1] for each continuous value)
    pred_scores = [min(list(classes), key=lambda x:abs(x-pred_raw[i])) for i in range(pred_raw.shape[0])]
    pred_scores = np.reshape(np.array(pred_scores), (-1, 7))
    save_with_pickle(pred_scores, pred_sc_save_name, model_folder)

    # Classes derived from predictions (useful for metrics including confusion matrix)
    pred_classes_dom = get_class_from_presence_score(pred_scores, predict_neutral_class, only_dominant=True)
    pred_classes_all = get_class_from_presence_score(pred_scores, predict_neutral_class, only_dominant=False)
    save_with_pickle(pred_classes_dom, pred_cl_dom_save_name, model_folder)
    save_with_pickle(pred_classes_all, pred_cl_all_save_name, model_folder)
    pred_raw = np.reshape(np.array(pred_raw), (-1, 7))
    # print(pred_raw[1,1])  # For debugging
    # print(pred_raw[:10, 1:])
    # print(pred_scores[:10, 1:])
    # print(pred_classes_all[:10, :])

    # Confusion matrix (binary classification: whether an emotion is present or not)
    num_classes = true_classes_all.shape[1]
    conf_matrix = multilabel_confusion_matrix(true_classes_all, pred_classes_all, labels=list(range(num_classes)))
    save_with_pickle(conf_matrix, 'conf_matrix_' + model_save_name, model_folder)
    # print("conf_matrix.shape", conf_matrix.shape)

    # # Model evaluation and prediction
    # print("\n\n================================= Model Evaluation ===========================================")
    # loss_function_val = model.evaluate(test_dataset, verbose=1)
    # print("Loss function ({}): {}".format(loss_function, loss_function_val))
    # # Regression metrics
    # mae = round(mean_absolute_error(true_scores, pred_raw), round_decimals)
    # mse = round(mean_squared_error(true_scores, pred_raw), round_decimals)
    # print("Mean absolute error:", mae)
    # print("Mean squared error:", mse)
    # # Classification metrics
    # acc = round(accuracy_score(true_classes, pred_classes), round_decimals)
    # acc_bal = round(balanced_accuracy_score(true_classes, pred_classes), round_decimals)
    # f1_each = f1_score(true_classes, pred_classes, average=None)
    # f1_macro = round(f1_score(true_classes, pred_classes, average='macro'), round_decimals)
    # f1_weighted = round(f1_score(true_classes, pred_classes, average='weighted'), round_decimals)
    # rec_each = recall_score(true_classes, pred_classes, average=None)
    # rec_macro = round(recall_score(true_classes, pred_classes, average='macro'), round_decimals)
    # rec_weighted = round(recall_score(true_classes, pred_classes, average='weighted'), round_decimals)
    # true_classes_bin = label_binarize(true_classes, classes=list(range(classes.shape[0])))
    # pred_classes_bin = label_binarize(pred_classes, classes=list(range(classes.shape[0])))
    # roc_auc_each = roc_auc_score(true_classes_bin, pred_classes_bin, average=None, multi_class='ovr')
    # roc_auc_macro = round(roc_auc_score(true_classes_bin, pred_classes_bin, average='macro', multi_class='ovr'),
    #                       round_decimals)
    # roc_auc_weighted = round(roc_auc_score(true_classes_bin, pred_classes_bin, average='weighted', multi_class='ovr'),
    #                          round_decimals)
    # print("Accuracy:", acc)
    # print("Balanced accuracy:", acc_bal)
    # print("F1 score (for each):", f1_each)
    # print("F1 score (unweighted mean):", f1_macro)
    # print("F1 score (weighted mean):", f1_weighted)
    # print("Recall (for each):", rec_each)
    # print("Recall (unweighted mean):", rec_macro)
    # print("Recall (weighted mean):", rec_weighted)
    # print("ROC AUC (for each):", roc_auc_each)
    # print("ROC AUC (unweighted mean):", roc_auc_macro)
    # print("ROC AUC (weighted mean):", roc_auc_weighted)
    #
    # save_results_in_csv_file(csv_name, csv_folder, num_layers, num_nodes, dropout_rate, batch_size, fixed_num_steps,
    #                          loss_function, loss_function_val, mae, mse, acc, acc_bal, f1_macro, f1_weighted,
    #                          rec_macro, rec_weighted, roc_auc_macro, roc_auc_weighted)
    # # TODO: Add metrics for each class
