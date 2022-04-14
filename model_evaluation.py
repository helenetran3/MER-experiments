import os
import numpy as np
import csv

from tensorflow.keras.models import load_model
from pickle_functions import save_with_pickle, pickle_file_exists, load_from_pickle, save_results_in_csv_file
from dataset_utils import get_tf_dataset

from sklearn.preprocessing import LabelEncoder, label_binarize, MultiLabelBinarizer
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score, recall_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score


def get_presence_score_from_finer_grained_val(pred_raw_emo, true_scores_all, coarse=False):
    """

    :param pred_raw_emo: array of shape (test_size, 6) predicting the 6 emotions
    :param true_scores_all: array of shape (test_size, 7) giving the true sentiment and the 6 emotions
    :param coarse: if True, the resulting presence scores in [0, 1, 2, 3].
                   Default: [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]
    :return: array of shape (test_size, 6) giving the presence score of all the 6 emotions
    """
    le = LabelEncoder()
    le.fit(true_scores_all[:, 1:].flatten())
    classes = le.classes_ if not coarse else [0, 1, 2, 3]
    pred_raw_emo = pred_raw_emo.flatten()
    pred_scores = [min(list(classes), key=lambda x: abs(x - pred_raw_emo[i])) for i in range(pred_raw_emo.shape[0])]
    pred_scores = np.reshape(np.array(pred_scores), (-1, 6))
    return pred_scores


def get_class_from_presence_score(score_array, with_neutral_class, only_dominant=False):
    """
    Get the dominant emotion class(es) for each test segment based on the presence scores of emotions.
    Column indexes: 0:happy, 1:sad, 2:anger, 3:surprise, 4:disgust, 5:fear, (6:neutral)

    :param score_array: array of shape (test_size,6) containing the presence score of 6 emotions
    :param with_neutral_class: whether we add a neutral class or not
    :param only_dominant: if True, only keep dominant classes (else keep any emotion present)
    :return: Binary array of shape (test_size,7) if we add neutral class, else array of shape (test_size,6).
    """

    list_emotions = []
    test_size = score_array.shape[0]

    if score_array.shape[1] != 6:
        print("Make sure that the array of presence scores have 6 columns (for the 6 emotions).")

    else:
        for seg_id in range(test_size):  # we check every segment

            score_seg = score_array[seg_id, :]
            max_score = max(score_seg)

            if max_score == 0:  # neutral case
                list_emotions.append([6]) if with_neutral_class else list_emotions.append([])

            else:  # there is at least one emotion
                if only_dominant:
                    # the following takes the index(es) of the maximum value in the presence score vector of emotions
                    list_emotions.append([i for i, val in enumerate(score_seg) if val == max_score])
                else:
                    list_emotions.append([i for i, val in enumerate(score_seg) if val != 0])

        num_classes = 7 if with_neutral_class else 6
        mlb = MultiLabelBinarizer(classes=list(range(num_classes)))
        bin_array_emotions = mlb.fit_transform(list_emotions)

        return bin_array_emotions


def compute_true_labels(y_test, model_folder, predict_neutral_class):

    true_sc_save_name = "true_scores_all"
    true_sc_coa_save_name = "true_scores_coarse"
    true_cl_pres_save_name = "true_classes_pres"
    true_cl_dom_save_name = "true_classes_dom"

    if not pickle_file_exists(true_sc_save_name, model_folder):

        # Compute true presence scores: arrays of shape (4654, 7)
        # Possible values: [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]
        # Coarse-grained values: [0, 1, 2, 3]
        # TODO remove sentiment prediction from model training and evaluation, and change the code consequently
        true_scores_all = np.reshape(np.array(y_test), (-1, 7))
        true_scores_all = true_scores_all[:, 1:]  # (4654, 6), removed the sentiment column  # TODO Reminder: Change here
        true_scores_coa = get_presence_score_from_finer_grained_val(true_scores_all, true_scores_all, coarse=True)
        save_with_pickle(true_scores_all, true_sc_save_name, model_folder)
        save_with_pickle(true_scores_coa, true_sc_coa_save_name, model_folder)

        # Compute true classes: binary arrays of shape (4654, 7)
        true_classes_pres = get_class_from_presence_score(true_scores_all, predict_neutral_class)
        true_classes_dom = get_class_from_presence_score(true_scores_all, predict_neutral_class, only_dominant=True)
        save_with_pickle(true_classes_pres, true_cl_pres_save_name, model_folder)
        save_with_pickle(true_classes_dom, true_cl_dom_save_name, model_folder)
        # print("true_scores_all[:10, 1:]:\n", true_scores_all[:10, 1:])
        # print("true_classes_pres[:10, :]:\n", true_classes_pres[:10, :])
        # print("true_classes_dom[:10, :]:\n", true_classes_dom[:10, :])

    else:
        true_scores_all = load_from_pickle(true_sc_save_name, model_folder)
        true_scores_coa = load_from_pickle(true_sc_coa_save_name, model_folder)
        true_classes_pres = load_from_pickle(true_cl_pres_save_name, model_folder)
        true_classes_dom = load_from_pickle(true_cl_dom_save_name, model_folder)

        return true_scores_all, true_scores_coa, true_classes_pres, true_classes_dom


def compute_pred_labels(pred_raw, parameters_name, true_scores_all, model_folder, predict_neutral_class):

    pred_sc_save_name = "pred_scores_all_{}".format(parameters_name)
    pred_sc_coa_save_name = "pred_scores_coarse_{}".format(parameters_name)
    pred_cl_pres_save_name = "pred_classes_pres_{}".format(parameters_name)
    pred_cl_dom_save_name = "pred_classes_dom_{}".format(parameters_name)

    # Get presence scores from raw predictions
    pred_raw_emo = pred_raw[:, 1:]  # (4654, 6), removed the sentiment column  # TODO Reminder: Change here
    pred_scores_all = get_presence_score_from_finer_grained_val(pred_raw_emo, true_scores_all)
    pred_scores_coa = get_presence_score_from_finer_grained_val(pred_raw_emo, true_scores_all, coarse=True)
    save_with_pickle(pred_scores_all, pred_sc_save_name, model_folder)
    save_with_pickle(pred_scores_coa, pred_sc_coa_save_name, model_folder)

    # Compute predicted classes from presence scores (useful for classification metrics including confusion matrix)
    pred_classes_pres = get_class_from_presence_score(pred_scores_all, predict_neutral_class)
    pred_classes_dom = get_class_from_presence_score(pred_scores_all, predict_neutral_class, only_dominant=True)
    save_with_pickle(pred_classes_pres, pred_cl_pres_save_name, model_folder)
    save_with_pickle(pred_classes_dom, pred_cl_dom_save_name, model_folder)
    # print("pred_raw_emo[:10, 1:]:\n", pred_raw_emo[:10, 1:])  # For debugging
    # print("pred_scores_all[:10, :]:\n", pred_scores_all[:10, :])
    # print("pred_scores_coa[:10, :]:\n", pred_scores_coa[:10, :])
    # print("pred_classes_pres[:10, :]:\n", pred_classes_pres[:10, :])
    # print("pred_classes_dom[:10, :]:\n", pred_classes_dom[:10, :])

    return pred_raw_emo, pred_scores_all, pred_scores_coa, pred_classes_pres, pred_classes_dom


def model_prediction(model, test_dataset, y_test, parameters_name, model_folder):

    print("\n\n================================= Model Prediction ===========================================")
    pred_raw_save_name = "pred_raw_{}".format(parameters_name)

    # Get raw score predictions from the model
    if not pickle_file_exists(pred_raw_save_name, model_folder):  # perform prediction (for debugging)
        num_test_samples = len(y_test)
        pred_raw = model.predict(test_dataset, verbose=1, steps=num_test_samples)  # (4654, 7)
        save_with_pickle(pred_raw, pred_raw_save_name, model_folder)
    else:
        pred_raw = load_from_pickle(pred_raw_save_name, model_folder)

    return pred_raw


def compute_loss_value(model, test_dataset, loss_function):

    print("\n\n================================= Model Evaluation ===========================================")

    loss_function_val = model.evaluate(test_dataset, verbose=1)
    print("Loss function ({}): {}".format(loss_function, loss_function_val))

    return loss_function_val


def get_regression_metrics(true_scores_all, pred_raw_emo, round_decimals):

    mae = round(mean_absolute_error(true_scores_all, pred_raw_emo), round_decimals)
    mse = round(mean_squared_error(true_scores_all, pred_raw_emo), round_decimals)
    print("Mean absolute error:", mae)
    print("Mean squared error:", mse)
    return [mae, mse]


def get_classification_metrics(true_classes, pred_classes, num_classes, round_decimals):
    """
    Compute classification metrics.
     Accuracy, balanced accuracy
     F1 for each emotion, unweighted and weighted F1
     Recall for each emotion, unweighted and weighted recall
     ROC AUC for each emotion, unweighted and weighted ROC AUC
    :param true_classes: binary array of true classes, shape (test_size, num_classes) (num_classes = 7 if with neutral class, else 6)
    :param pred_classes: binary array of predictions, shape (test_size, num_classes) (num_classes = 7 if with neutral class, else 6)
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param round_decimals: number of decimals to be rounded for metrics
    :return:
    """

    acc = round(accuracy_score(true_classes, pred_classes), round_decimals)
    f1_each = f1_score(true_classes, pred_classes, average=None)
    f1_macro = round(f1_score(true_classes, pred_classes, average='macro'), round_decimals)
    f1_weighted = round(f1_score(true_classes, pred_classes, average='weighted'), round_decimals)
    rec_each = recall_score(true_classes, pred_classes, average=None)
    rec_macro = round(recall_score(true_classes, pred_classes, average='macro'), round_decimals)
    rec_weighted = round(recall_score(true_classes, pred_classes, average='weighted'), round_decimals)
    true_classes_bin = label_binarize(true_classes, classes=list(range(num_classes)))
    pred_classes_bin = label_binarize(pred_classes, classes=list(range(num_classes)))
    roc_auc_each = roc_auc_score(true_classes_bin, pred_classes_bin, average=None, multi_class='ovr')
    roc_auc_macro = round(roc_auc_score(true_classes_bin, pred_classes_bin, average='macro', multi_class='ovr'),
                          round_decimals)
    roc_auc_weighted = round(roc_auc_score(true_classes_bin, pred_classes_bin, average='weighted', multi_class='ovr'),
                             round_decimals)

    f1_each_rounded = [round(val, round_decimals) for val in f1_each]
    rec_each_rounded = [round(val, round_decimals) for val in rec_each]
    roc_auc_each_rounded = [round(val, round_decimals) for val in roc_auc_each]

    print("Multilabel Accuracy:", acc)
    print("F1 score (for each):", f1_each_rounded)
    print("F1 score (unweighted mean):", f1_macro)
    print("F1 score (weighted mean):", f1_weighted)
    print("Recall (for each):", rec_each_rounded)
    print("Recall (unweighted mean):", rec_macro)
    print("Recall (weighted mean):", rec_weighted)
    print("ROC AUC (for each):", roc_auc_each_rounded)
    print("ROC AUC (unweighted mean):", roc_auc_macro)
    print("ROC AUC (weighted mean):", roc_auc_weighted)

    res = [acc, f1_macro, f1_weighted, rec_macro, rec_weighted, roc_auc_macro, roc_auc_weighted]
    res = res + f1_each_rounded + rec_each_rounded + roc_auc_each_rounded

    return res


def evaluate_model(test_list, batch_size, fixed_num_steps, num_layers, num_nodes, dropout_rate, loss_function,
                   model_name, predict_neutral_class, round_decimals):
    """
    Evaluate the performance of the best model.

    :param test_list: [x_test, y_test, seg_test]
    :param batch_size: Batch size for training
    :param fixed_num_steps: Fixed size for all the sequences (if we keep the original size, this parameter is set to 0)
    :param num_layers: Number of bidirectional layers for the model
    :param num_nodes: Number of nodes for the penultimate dense layer
    :param dropout_rate: Dropout rate before each dense layer
    :param loss_function: Loss function
    :param model_name: Name of the model currently tested
    :param predict_neutral_class: Whether we predict the neutral class
    :param round_decimals: Number of decimals to be rounded for metrics
    """

    # Load best model
    parameters_name = "l_{}_n_{}_d_{}_b_{}_s_{}".format(num_layers, num_nodes, dropout_rate, batch_size, fixed_num_steps)
    model_save_name = "model_{}.h5".format(parameters_name)
    model_folder = os.path.join('models', model_name)
    model_save_path = os.path.join(model_folder, model_save_name)
    model = load_model(model_save_path)

    # Extract x, y and seg_ids for test set
    x_test = test_list[0]  # each element of shape (29, 409)
    y_test = test_list[1]  # each element of shape (1, 7)
    seg_test = test_list[2]

    # Create TensorFlow test dataset for model prediction and evaluation
    with_fixed_length = (fixed_num_steps > 0)
    test_dataset = get_tf_dataset(x_test, y_test, seg_test, batch_size, with_fixed_length, fixed_num_steps,
                                  train_mode=False)

    # True labels
    true_scores_all, true_scores_coa, true_classes_pres, true_classes_dom = compute_true_labels(y_test, model_folder,
                                                                                                predict_neutral_class)

    # Predicted labels
    pred_raw = model_prediction(model, test_dataset, y_test, parameters_name, model_folder)
    pred_raw_emo, pred_scores, pred_scores_coa, \
    pred_classes_pres, pred_classes_dom = compute_pred_labels(pred_raw, parameters_name, true_scores_all, model_folder,
                                                              predict_neutral_class)

    # Confusion matrix (binary classification: whether an emotion is present or not)
    num_classes = true_classes_pres.shape[1]
    conf_matrix = multilabel_confusion_matrix(true_classes_pres, pred_classes_pres, labels=list(range(num_classes)))
    save_with_pickle(conf_matrix, 'conf_matrix_{}'.format(parameters_name), model_folder)

    # Model evaluation
    loss_function_val = compute_loss_value(model, test_dataset, loss_function)

    # Regression metrics
    metrics_regression = get_regression_metrics(true_scores_all, pred_raw_emo, round_decimals)

    ## TODO Quatre cas: les scores de présence par défaut, 4 scores de présence, présence ou absence d'une émotion et
    ## TODO             classification de le ou les émotions dominantes -> utile pour l'ambiguité
    # Classification metrics
    print("\n------ Presence/absence of an emotion ------")
    metrics_presence = get_classification_metrics(true_classes_pres, pred_classes_pres, num_classes, round_decimals)
    print("\n----- Prediction of a dominant emotion -----")
    metrics_dominant = get_classification_metrics(true_classes_dom, pred_classes_dom, num_classes, round_decimals)

    save_results_in_csv_file(model_name, num_layers, num_nodes, dropout_rate, batch_size, fixed_num_steps,
                             loss_function, loss_function_val, metrics_regression, metrics_presence, metrics_dominant,
                             predict_neutral_class)
