import numpy as np

from src.pickle_functions import save_with_pickle

from sklearn.preprocessing import label_binarize
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, multilabel_confusion_matrix


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


def get_regression_metrics(true_scores_all, pred_scores_all, round_decimals):
    """
    Compute regression metrics: mean absolute error and mean squared error

    :param true_scores_all: array of shape (test size, 6) giving the true scores for the 6 emotions (given by the database)
    :param pred_scores_all: array of shape (test_size, 6) predicting the presence score of the 6 emotions
    :param round_decimals: number of decimals to be rounded for metrics
    :return: List of classification metrics
    """

    num_emotions = true_scores_all.shape[1]
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
    prec_weighted = round(precision_score(true_classes, pred_classes, average='weighted', zero_division=1),
                          round_decimals)
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
