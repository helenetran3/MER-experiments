import os
import numpy as np

from tensorflow.keras.models import load_model
from src.pickle_functions import save_with_pickle, pickle_file_exists, load_from_pickle
from src.csv_functions import save_results_in_csv_file
from src.dataset_utils import get_tf_dataset

from sklearn.preprocessing import LabelEncoder, label_binarize, MultiLabelBinarizer
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def create_array_true_scores(y_test, num_classes, model_folder):
    """
    Return the true scores given by the database.
    Possible values: [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]

    :param y_test: List of arrays of shape (1, num_classes)
    :param num_classes: Number of classes
    :param model_folder: Name of the folder where the true_scores_all will be saved
    :return: true_scores_all
    """

    if not pickle_file_exists("true_scores_all", root_folder=model_folder):
        true_scores_all = np.reshape(np.array(y_test), (-1, num_classes))
        save_with_pickle(true_scores_all, "true_scores_all", root_folder=model_folder)
    else:
        true_scores_all = load_from_pickle("true_scores_all", root_folder=model_folder)
    return true_scores_all


def get_presence_score_from_finer_grained_val(pred_raw, true_scores_all, num_classes, coarse=False):
    """
    Get the presence score from a finer grained presence score
    (raw predictions -> 12 or 4 presence scores, or 12 presence scores -> 4 presence scores)

    :param pred_raw: array of shape (test_size, num_classes) predicting the presence score of the 6 emotions
    :param true_scores_all: array of shape (test_size, num_classes) giving the presence score of the 6 emotions
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param coarse: if True, the resulting presence scores in [0, 1, 2, 3].
                   Default: [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]
    :return: array of shape (test_size, 6) giving the presence score of all the 6 emotions
    """

    if not coarse:
        le = LabelEncoder()
        le.fit(true_scores_all.flatten())
        classes = le.classes_
    else:
        classes = [0, 1, 2, 3]
    pred_raw = pred_raw.flatten()
    pred_scores = [min(list(classes), key=lambda x: abs(x - pred_raw[i])) for i in range(pred_raw.shape[0])]
    pred_scores = np.reshape(np.array(pred_scores), (-1, num_classes))
    return pred_scores


def get_class_from_presence_score(score_array, with_neutral_class, threshold_emo_pres, num_classes, only_dominant=False):
    """
    Get the dominant emotion class(es) for each test segment based on the presence scores of emotions.
    If only_dominant, return a binary array, else return a list of 7 binary arrays for the 7 thresholds [0, 0.5, 1, 1.5, 2, 2.5, 3].
    Column indexes: 0:happy, 1:sad, 2:anger, 3:surprise, 4:disgust, 5:fear, (6:neutral)

    :param score_array: array of shape (test_size, num_classes) containing the presence score of 6 emotions (+ neutral)
    :param with_neutral_class: whether we add a neutral class or not
    :param threshold_emo_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param only_dominant: if True, only keep dominant classes (else keep any emotion present)
    :return: Binary array of shape (test_size,7) if we add neutral class, else array of shape (test_size, 6).
    """

    def get_list_emotions_for_each_seg(score_array, test_size, with_neutral_class, only_dominant, thres):
        list_emotions = []

        for seg_id in range(test_size):  # we check every segment

            score_seg = score_array[seg_id, :]
            max_score = max(score_seg)

            if max_score == 0:  # the current segment shows neutral (no emotions)
                list_emotions.append([6]) if with_neutral_class else list_emotions.append([])

            else:  # there is at least one emotion
                if only_dominant:
                    # the following takes the index(es) of the maximum value in the presence score vector of emotions
                    list_emotions.append([i for i, val in enumerate(score_seg) if val == max_score])
                else:
                    tmp = [i for i, val in enumerate(score_seg) if val > 0] if thres == 0 else \
                        [i for i, val in enumerate(score_seg) if val >= thres]
                    tmp = [6] if len(tmp) == 0 and with_neutral_class else tmp
                    list_emotions.append(tmp)

        return list_emotions

    test_size = score_array.shape[0]
    mlb = MultiLabelBinarizer(classes=list(range(num_classes)))

    if only_dominant:
        list_emotions = get_list_emotions_for_each_seg(score_array, test_size, with_neutral_class,
                                                       only_dominant=True, thres=None)
        bin_array_emotions = mlb.fit_transform(list_emotions)

        return bin_array_emotions

    else:  # Emotions present
        list_emotions_all_thresholds = []

        for thres in threshold_emo_pres:
            list_emotions_thres = get_list_emotions_for_each_seg(score_array, test_size, with_neutral_class,
                                                                 only_dominant=False, thres=thres)
            list_emotions_all_thresholds.append(list_emotions_thres)

        list_bin_array_emotions = [mlb.fit_transform(l) for l in list_emotions_all_thresholds]

        return list_bin_array_emotions


def compute_true_labels(true_scores_all, predict_neutral_class, threshold_emo_pres, num_classes, model_folder):
    """
    Compute the true labels (all arrays of shape (test_size, 6), true_classes_pres is a list of arrays of this type)
    true_scores_coa: Closest true score among [0, 1, 2, 3]
    true_classes_pres: Emotions truly present (varying thresholds for presence score given by threshold_emo_pres)
    true_classes_dom: Emotions truly dominant (highest presence score)

    :param true_scores_all: array of shape (test size, 6) giving the true scores for the 6 emotions (given by the database).
           Possible values of true_scores_all: [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]
    :param predict_neutral_class: Whether we predict the neutral class
    :param threshold_emo_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param model_folder: The folder where the results will be saved
    :return: true_scores_coa, true_classes_pres, true_classes_dom
    """

    if not pickle_file_exists("true_scores_coarse", root_folder=model_folder):

        # Compute true presence scores: arrays of shape (4654, 7)
        # Possible values: [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]
        # Coarse-grained values: [0, 1, 2, 3]
        true_scores_coa = get_presence_score_from_finer_grained_val(true_scores_all, true_scores_all, num_classes, coarse=True)
        save_with_pickle(true_scores_coa, "true_scores_coarse", root_folder=model_folder)

        # Compute true classes: binary arrays of shape (4654, 6 or 7)
        true_classes_pres = get_class_from_presence_score(true_scores_all, predict_neutral_class, threshold_emo_pres,
                                                          num_classes)
        true_classes_dom = get_class_from_presence_score(true_scores_all, predict_neutral_class, threshold_emo_pres,
                                                         num_classes, only_dominant=True)
        save_with_pickle(true_classes_pres, "true_classes_pres", root_folder=model_folder)
        save_with_pickle(true_classes_dom, "true_classes_dom", root_folder=model_folder)

    else:
        true_scores_coa = load_from_pickle("true_scores_coarse", root_folder=model_folder)
        true_classes_pres = load_from_pickle("true_classes_pres", root_folder=model_folder)
        true_classes_dom = load_from_pickle("true_classes_dom", root_folder=model_folder)

    return true_scores_coa, true_classes_pres, true_classes_dom


def compute_pred_labels(pred_raw, true_scores_all, predict_neutral_class, threshold_emo_pres, parameters_name,
                        num_classes, model_folder):
    """
    Compute the prediction labels (all arrays of shape (test_size, 6), pred_classes_pres is a list of arrays of this type))
    pred_scores_all: Closest predicted score among [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]
    pred_scores_coa: Closest predicted score among [0, 1, 2, 3]
    pred_classes_pres: Emotions present predicted by the model (varying thresholds for presence score given by threshold_emo_pres)
    pred_classes_dom: Dominant emotions predicted by the model (highest presence score)

    :param pred_raw: array of shape (test_size, 7) predicting the presence score of the 6 emotions
    :param true_scores_all: array of shape (test size, 6) giving the true scores for the 6 emotions (given by the database)
    :param predict_neutral_class: Whether we predict the neutral class
    :param threshold_emo_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param parameters_name: String describing the model training parameters (used to append to the pickle object name)
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param model_folder: The folder where the results will be saved
    :return: pred_scores_all, pred_scores_coa, pred_classes_pres, pred_classes_dom
    """

    pred_sc_save_name = "pred_scores_all_{}".format(parameters_name)
    pred_sc_coa_save_name = "pred_scores_coarse_{}".format(parameters_name)
    pred_cl_pres_save_name = "pred_classes_pres_{}".format(parameters_name)
    pred_cl_dom_save_name = "pred_classes_dom_{}".format(parameters_name)

    # Get presence scores from raw predictions
    pred_scores_all = get_presence_score_from_finer_grained_val(pred_raw, true_scores_all, num_classes)
    pred_scores_coa = get_presence_score_from_finer_grained_val(pred_raw, true_scores_all, num_classes, coarse=True)
    save_with_pickle(pred_scores_all, pred_sc_save_name, root_folder=model_folder)
    save_with_pickle(pred_scores_coa, pred_sc_coa_save_name, root_folder=model_folder)

    # Compute predicted classes from presence scores (useful for classification metrics including confusion matrix)
    pred_classes_pres = get_class_from_presence_score(pred_scores_all, predict_neutral_class, threshold_emo_pres,
                                                      num_classes)
    pred_classes_dom = get_class_from_presence_score(pred_scores_all, predict_neutral_class, threshold_emo_pres,
                                                     num_classes, only_dominant=True)
    save_with_pickle(pred_classes_pres, pred_cl_pres_save_name, root_folder=model_folder)
    save_with_pickle(pred_classes_dom, pred_cl_dom_save_name, root_folder=model_folder)

    return pred_scores_all, pred_scores_coa, pred_classes_pres, pred_classes_dom


def get_binary_arrays_from_scores_coa(scores_coa, num_classes):
    """
    Returns a list of num_classes binary arrays for each emotion. Each array has shape (test_size, 4).
    Useful for classification metrics from coarse presence scores.

    :param scores_coa: array of size (test_size, num_classes) giving the coarse presence scores (0, 1, 2, or 3)
    :param num_classes: number of classes (7 with neutral class, else 6)
    :return: The list of binary arrays
    """

    scores_for_each_emo = [scores_coa[:, i] for i in range(num_classes)]
    bin_arr_for_each_emo = [label_binarize(sc, classes=range(4)) for sc in scores_for_each_emo]

    return bin_arr_for_each_emo


def model_prediction(model, test_dataset, num_test_samples, parameters_name, model_folder):
    """
    Return the predictions of the model on test dataset.

    :param model: The model to evaluate
    :param test_dataset: The TensorFlow dataset for test set
    :param num_test_samples: Number of test samples
    :param parameters_name: String describing the model training parameters (used to append to the pickle object name)
    :param model_folder: The folder where the result will be saved
    :return: pred_raw: array giving the predictions of the model on test dataset.
    """

    pred_raw_save_name = "pred_raw_{}".format(parameters_name)

    # Get raw score predictions from the model
    if not pickle_file_exists(pred_raw_save_name, root_folder=model_folder):  # perform prediction (for debugging)
        print("\n\n================================= Model Prediction ===========================================\n")
        pred_raw = model.predict(test_dataset, verbose=1, steps=num_test_samples)  # (4654, num_classes)
        save_with_pickle(pred_raw, pred_raw_save_name, root_folder=model_folder)
    else:
        pred_raw = load_from_pickle(pred_raw_save_name, root_folder=model_folder)

    return pred_raw


def compute_multilabel_confusion_matrix(true_classes_pres, pred_classes_pres, threshold_emo_pres, num_classes,
                                        parameters_name, model_folder):
    """
    Compute the multilabel confusion matrix for each emotion (whether the emotion is present or not).
    Pickle file generated.

    :param true_classes_pres: List of arrays giving the emotions truly present (varying thresholds for presence score given by threshold_emo_pres)
    :param pred_classes_pres: List of arrays giving the emotions present predicted by the model (varying thresholds)
    :param threshold_emo_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param parameters_name: String describing the model training parameters (used to append to the pickle object name)
    :param model_folder: The folder where the results will be saved
    :return:
    """

    for i, thres in enumerate(threshold_emo_pres):
        conf_matrix = multilabel_confusion_matrix(true_classes_pres[i], pred_classes_pres[i],
                                                  labels=list(range(num_classes)))
        save_with_pickle(conf_matrix, 'conf_matrix_t_{}_{}'.format(thres, parameters_name), root_folder=model_folder)


def compute_loss_value(model, test_dataset, loss_function, round_decimals):
    """
    Return the loss function value of the model on test dataset.

    :param model: The model to evaluate
    :param test_dataset: The TensorFlow dataset for test set
    :param loss_function: Name of the loss function
    :param round_decimals: number of decimals to be rounded for metrics
    :return: The loss function value
    """

    print("\n\n================================= Model Evaluation ===========================================\n")

    loss_function_val = model.evaluate(test_dataset, verbose=1)
    loss_function_val = round(loss_function_val, round_decimals)
    print("Loss function ({}): {}".format(loss_function, loss_function_val))

    return loss_function_val


def get_regression_metrics(true_scores_all, pred_scores_all, round_decimals):
    """
    Compute regression metrics: mean absolute error and mean squared error

    :param true_scores_all: array of shape (test size, 6) giving the true scores for the 6 emotions (given by the database)
    :param pred_scores_all: array of shape (test_size, 6) predicting the presence score of the 6 emotions
    :param round_decimals: number of decimals to be rounded for metrics
    :return: List of classification metrics
    """

    num_emotions = true_scores_all.shape[1]

    if num_emotions != 6:
        print("Make sure that the array of presence scores have 6 columns (for the 6 emotions).")

    true_scores_emo = [true_scores_all[:, i] for i in range(num_emotions)]
    pred_scores_emo = [pred_scores_all[:, i] for i in range(num_emotions)]

    mae_emo = [round(mean_absolute_error(t, p), round_decimals) for t, p in zip(true_scores_emo, pred_scores_emo)]
    mse_emo = [round(mean_squared_error(t, p), round_decimals) for t, p in zip(true_scores_emo, pred_scores_emo)]
    r2_emo = [round(r2_score(t, p), round_decimals) for t, p in zip(true_scores_emo, pred_scores_emo)]
    mae = round(mean_absolute_error(true_scores_all, pred_scores_all), round_decimals)
    mse = round(mean_squared_error(true_scores_all, pred_scores_all), round_decimals)
    r2 = round(r2_score(true_scores_all, pred_scores_all), round_decimals)

    print("Mean absolute error (for each emotion):", mae_emo)
    print("Mean squared error (for each emotion):", mse_emo)
    print("R2 score (for each emotion):", r2_emo)
    print("Mean absolute error (overall):", mae)
    print("Mean squared error (overall):", mse)
    print("R2 score (overall):", r2)

    return [mae, mse, r2] + mae_emo + mse_emo + r2_emo


def get_classification_metrics_score_coa(true_scores_coa, pred_scores_coa, num_classes, round_decimals):
    """
    Return unweighted mean of all the classification metrics for the case of coarse scores.

    :param true_scores_coa: Array of shape (test_size, 6) giving the true presence scores among [0, 1, 2, 3]
    :param pred_scores_coa: Array of shape (test_size, 6) giving the predicted presence scores among [0, 1, 2, 3]
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param round_decimals: number of decimals to be rounded for metrics
    :return: [f1_macro_uw_mean, f1_weighted_uw_mean, rec_macro_uw_mean, rec_weighted_uw_mean, prec_macro_uw_mean,
              prec_weighted_uw_mean, roc_auc_macro_uw_mean, roc_auc_weighted_uw_mean]
    """

    true_scores_bin = get_binary_arrays_from_scores_coa(true_scores_coa, num_classes)
    pred_scores_bin = get_binary_arrays_from_scores_coa(pred_scores_coa, num_classes)
    true_scores_bin_stack = np.vstack(true_scores_bin)
    pred_scores_bin_stack = np.vstack(pred_scores_bin)

    # In this block, we go through all the emotions. Metrics are here given for each emotion.
    # Imbalance in presence score [0, 1, 2, 3]. example for happy: [3048, 1210, 376, 20],
    # hence we also calculate weighted metrics
    f1_macro_each = [round(f1_score(ts, ps, average='macro', zero_division=1), round_decimals)
                    for ts, ps in zip(true_scores_bin, pred_scores_bin)]
    f1_weighted_each = [round(f1_score(ts, ps, average='weighted', zero_division=1), round_decimals)
                       for ts, ps in zip(true_scores_bin, pred_scores_bin)]
    rec_macro_each = [round(recall_score(ts, ps, average='macro', zero_division=1), round_decimals)
                     for ts, ps in zip(true_scores_bin, pred_scores_bin)]
    rec_weighted_each = [round(recall_score(ts, ps, average='weighted', zero_division=1), round_decimals)
                        for ts, ps in zip(true_scores_bin, pred_scores_bin)]
    prec_macro_each = [round(precision_score(ts, ps, average='macro', zero_division=1), round_decimals)
                      for ts, ps in zip(true_scores_bin, pred_scores_bin)]
    prec_weighted_each = [round(precision_score(ts, ps, average='weighted', zero_division=1), round_decimals)
                         for ts, ps in zip(true_scores_bin, pred_scores_bin)]
    # roc_auc_macro_each = [roc_auc_score(ts, ps, average='macro')
    #                      for ts, ps in zip(true_scores_bin, pred_scores_bin)]
    # roc_auc_weighted_each = [roc_auc_score(ts, ps, average='weighted')
    #                         for ts, ps in zip(true_scores_bin, pred_scores_bin)]

    # Unweighted mean over the six emotions
    acc = round(accuracy_score(true_scores_bin_stack, pred_scores_bin_stack), round_decimals)
    f1_macro_uw_mean = round(np.mean(f1_macro_each), round_decimals)
    f1_weighted_uw_mean = round(np.mean(f1_weighted_each), round_decimals)
    rec_macro_uw_mean = round(np.mean(rec_macro_each), round_decimals)
    rec_weighted_uw_mean = round(np.mean(rec_weighted_each), round_decimals)
    prec_macro_uw_mean = round(np.mean(prec_macro_each), round_decimals)
    prec_weighted_uw_mean = round(np.mean(prec_weighted_each), round_decimals)
    # roc_auc_macro_uw_mean = round(np.mean(roc_auc_macro_each), round_decimals)
    # roc_auc_weighted_uw_mean = round(np.mean(roc_auc_weighted_each), round_decimals)
    print("4 classes (presence scores) for each emotion class. "
          "The following metrics all give the unweighted mean over the emotions and the brackets give the type "
          "of mean over the presence scores.\n")
    print("Accuracy:", acc)
    print("Unweighted F1 score (for each emotion):", f1_macro_each)
    print("Weighted F1 score (for each emotion):", f1_weighted_each)
    print("Unweighted recall score (for each emotion):", rec_macro_each)
    print("Weighted recall score (for each emotion):", rec_weighted_each)
    print("Unweighted precision score (for each emotion):", prec_macro_each)
    print("Weighted precision score (for each emotion):", prec_weighted_each)
    print("F1 score (weighted mean):", f1_macro_uw_mean)
    print("F1 score (unweighted mean):", f1_weighted_uw_mean)
    print("Recall (weighted mean):", rec_macro_uw_mean)
    print("Recall (unweighted mean):", rec_weighted_uw_mean)
    print("Precision (weighted mean):", prec_macro_uw_mean)
    print("Precision (unweighted mean):", prec_weighted_uw_mean)
    # print("ROC AUC (weighted mean):", roc_auc_macro_uw_mean)
    # print("ROC AUC (unweighted mean):", roc_auc_weighted_uw_mean)

    res = [acc, f1_macro_uw_mean, f1_weighted_uw_mean, rec_macro_uw_mean, rec_weighted_uw_mean, prec_macro_uw_mean,
           prec_weighted_uw_mean]  # , roc_auc_macro_uw_mean, roc_auc_weighted_uw_mean]

    res = res + f1_macro_each + f1_weighted_each + rec_macro_each + rec_weighted_each + prec_macro_each \
          + prec_weighted_each  # + roc_auc_macro_each + roc_auc_weighted_each

    return res


def get_classification_metrics(true_classes, pred_classes, num_classes, round_decimals, thres=None):
    """
    Compute classification metrics.
     Accuracy
     F1 for each emotion, unweighted and weighted F1
     Recall for each emotion, unweighted and weighted recall
     Precision for each emotion, unweighted and weighted precision
     ROC AUC for each emotion, unweighted and weighted ROC AUC
    :param true_classes: binary array of true classes, shape (test_size, num_classes) (num_classes = 7 if with neutral class, else 6)
    :param pred_classes: binary array of predictions, shape (test_size, num_classes) (num_classes = 7 if with neutral class, else 6)
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param round_decimals: number of decimals to be rounded for metrics
    :return: List of classification metrics
    """

    acc = round(accuracy_score(true_classes, pred_classes), round_decimals)
    f1_each = f1_score(true_classes, pred_classes, average=None, zero_division=1)
    f1_macro = round(f1_score(true_classes, pred_classes, average='macro', zero_division=1), round_decimals)
    f1_weighted = round(f1_score(true_classes, pred_classes, average='weighted', zero_division=1), round_decimals)
    rec_each = recall_score(true_classes, pred_classes, average=None, zero_division=1)
    rec_macro = round(recall_score(true_classes, pred_classes, average='macro', zero_division=1), round_decimals)
    rec_weighted = round(recall_score(true_classes, pred_classes, average='weighted', zero_division=1), round_decimals)
    prec_each = precision_score(true_classes, pred_classes, average=None, zero_division=1)
    prec_macro = round(precision_score(true_classes, pred_classes, average='macro', zero_division=1), round_decimals)
    prec_weighted = round(precision_score(true_classes, pred_classes, average='weighted', zero_division=1), round_decimals)
    true_classes_bin = label_binarize(true_classes, classes=list(range(num_classes)))
    pred_classes_bin = label_binarize(pred_classes, classes=list(range(num_classes)))
    # Next block: Compute ROC AUC when NOT defining the presence/absence of an emotion (as this task creates error
    # when no emotion has a presence score greater than the threshold - ROC AUC needs at least one positive sample)
    if thres is None:
        roc_auc_each = roc_auc_score(true_classes_bin, pred_classes_bin, average=None, multi_class='ovr')
        roc_auc_macro = round(roc_auc_score(true_classes_bin, pred_classes_bin, average='macro', multi_class='ovr'),
                              round_decimals)
        roc_auc_weighted = round(
            roc_auc_score(true_classes_bin, pred_classes_bin, average='weighted', multi_class='ovr'), round_decimals)
        roc_auc_each_rounded = [round(val, round_decimals) for val in roc_auc_each]

    f1_each_rounded = [round(val, round_decimals) for val in f1_each]
    rec_each_rounded = [round(val, round_decimals) for val in rec_each]
    prec_each_rounded = [round(val, round_decimals) for val in prec_each]

    if thres is not None:
        print('\n>>> Threshold:', thres)

    print("Multilabel Accuracy:", acc)
    print("F1 score (for each):", f1_each_rounded)
    print("F1 score (unweighted mean):", f1_macro)
    print("F1 score (weighted mean):", f1_weighted)
    print("Recall (for each):", rec_each_rounded)
    print("Recall (unweighted mean):", rec_macro)
    print("Recall (weighted mean):", rec_weighted)
    print("Precision (for each):", prec_each_rounded)
    print("Precision (unweighted mean):", prec_macro)
    print("Precision (weighted mean):", prec_weighted)
    if thres is None:
        print("ROC AUC (for each):", roc_auc_each_rounded)
        print("ROC AUC (unweighted mean):", roc_auc_macro)
        print("ROC AUC (weighted mean):", roc_auc_weighted)

    res = [acc, f1_macro, f1_weighted, rec_macro, rec_weighted, prec_macro, prec_weighted]
    if thres is None:
        res = res + [roc_auc_macro, roc_auc_weighted] + f1_each_rounded + rec_each_rounded + prec_each_rounded + \
              roc_auc_each_rounded
    else:
        res = res + f1_each_rounded + rec_each_rounded + prec_each_rounded

    return res


def evaluate_model(test_list, batch_size, fixed_num_steps, num_layers, num_nodes, dropout_rate, loss_function,
                   model_name, predict_neutral_class, threshold_emo_pres, round_decimals):
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
    :param threshold_emo_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param round_decimals: Number of decimals to be rounded for metrics
    """

    # Load best model
    parameters_name = "l_{}_n_{}_d_{}_b_{}_s_{}".format(num_layers, num_nodes, dropout_rate, batch_size,
                                                        fixed_num_steps)
    model_save_name = "model_{}.h5".format(parameters_name)
    model_folder = os.path.join('models_tested', model_name)
    model_save_path = os.path.join(model_folder, model_save_name)
    model = load_model(model_save_path)

    # Extract x, y and seg_ids for test set
    x_test = test_list[0]  # each element of shape (29, 409)
    y_test = test_list[1]  # each element of shape (1, 6 or 7)
    seg_test = test_list[2]
    num_classes = y_test[0].shape[1]
    num_test_samples = len(y_test)

    # Create TensorFlow test dataset for model prediction and evaluation
    with_fixed_length = (fixed_num_steps > 0)
    test_dataset = get_tf_dataset(x_test, y_test, seg_test, num_classes, batch_size, with_fixed_length, fixed_num_steps,
                                  train_mode=False)

    # True labels
    true_scores_all = create_array_true_scores(y_test, num_classes, model_folder)
    true_scores_coa, true_classes_pres, true_classes_dom = compute_true_labels(true_scores_all,
                                                                               predict_neutral_class,
                                                                               threshold_emo_pres, num_classes,
                                                                               model_folder)

    # Predicted labels
    pred_raw = model_prediction(model, test_dataset, num_test_samples, parameters_name, model_folder)
    pred_scores, pred_scores_coa, pred_classes_pres, pred_classes_dom = compute_pred_labels(pred_raw, true_scores_all,
                                                                                            predict_neutral_class,
                                                                                            threshold_emo_pres,
                                                                                            parameters_name,
                                                                                            num_classes,
                                                                                            model_folder)

    # Confusion matrix
    compute_multilabel_confusion_matrix(true_classes_pres, pred_classes_pres, threshold_emo_pres, num_classes,
                                        parameters_name, model_folder)

    # Model evaluation
    loss_function_val = compute_loss_value(model, test_dataset, loss_function, round_decimals)

    print("\n\n")
    print("In the following, all the metrics displayed for each emotion are in this order: sentiment, happy, sad, "
          "anger, surprise, disgust, fear", end="")
    print(", neutral.\n") if predict_neutral_class else print(".\n")

    # Regression metrics
    print("\n------- Presence score estimation (regression) --------\n")
    metrics_regression = get_regression_metrics(true_scores_all, pred_raw, round_decimals)

    # Classification metrics
    print("\n------- Presence score classification [0,1,2,3] -------\n")
    metrics_score_coa = get_classification_metrics_score_coa(true_scores_coa, pred_scores_coa, num_classes,
                                                             round_decimals)

    print("\n------- Detecting the presence of emotions ------------")
    metrics_presence = [get_classification_metrics(true_classes_pres[i], pred_classes_pres[i], num_classes,
                                                   round_decimals, thres) for i, thres in enumerate(threshold_emo_pres)]
    print("\n------- Predicting dominant emotions ------------------\n")
    metrics_dominant = get_classification_metrics(true_classes_dom, pred_classes_dom, num_classes, round_decimals)

    save_results_in_csv_file(model_name, num_layers, num_nodes, dropout_rate, batch_size, fixed_num_steps,
                             loss_function, loss_function_val, metrics_regression, metrics_score_coa, metrics_presence,
                             metrics_dominant, predict_neutral_class, threshold_emo_pres)
