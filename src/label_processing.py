import numpy as np

from src.pickle_functions import *

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer


def get_available_presence_scores(true_scores_all):
    le = LabelEncoder()
    le.fit(true_scores_all.flatten())
    return le.classes_


def get_presence_score_from_finer_grained_val(pred_raw, all_scores, num_classes, coarse=False):
    """
    Get the presence score from a finer grained presence score
    (raw predictions -> 12 or 4 presence scores, or 12 presence scores -> 4 presence scores)

    :param pred_raw: array of shape (test_size, num_classes) predicting the presence score of the 6 emotions
    :param all_scores: list of available presence scores
    :param num_classes: number of emotion classes (7 with neutral class, else 6)
    :param coarse: if True, the resulting presence scores in [0, 1, 2, 3].
                   Default: [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]
    :return: array of shape (test_size, 6) giving the presence score of all the 6 emotions
    """

    classes = all_scores if not coarse else [0, 1, 2, 3]
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

    def get_list_emotions_for_each_seg(thres=None):
        list_emotions = []
        test_size = score_array.shape[0]

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

    mlb = MultiLabelBinarizer(classes=list(range(num_classes)))

    if only_dominant:
        list_emotions = get_list_emotions_for_each_seg()
        bin_array_emotions = mlb.fit_transform(list_emotions)
        return bin_array_emotions

    else:  # Collect all emotions present in the sample
        list_emotions_all_thresholds = []

        for thres in threshold_emo_pres:
            list_emotions_thres = get_list_emotions_for_each_seg(thres=thres)
            list_emotions_all_thresholds.append(list_emotions_thres)

        list_bin_array_emotions = [mlb.fit_transform(l) for l in list_emotions_all_thresholds]
        return list_bin_array_emotions


def create_array_true_scores(y_test, num_classes, ext_name):
    """
    Return the true scores given by the database.
    Possible values: [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]

    :param y_test: List of arrays of shape (1, num_classes)
    :param num_classes: Number of classes
    :param ext_name: extension name containing info on whether we predict sentiment/neutral class
    :return: true_scores_all
    """

    if not pickle_file_exists("true_scores_all" + ext_name, pickle_folder='processed_folds', root_folder='cmu_mosei'):
        true_scores_all = np.reshape(np.array(y_test), (-1, num_classes))
        save_with_pickle(true_scores_all, "true_scores_all" + ext_name,
                         pickle_folder='processed_folds', root_folder='cmu_mosei')
    else:
        true_scores_all = load_from_pickle("true_scores_all" + ext_name,
                                           pickle_folder='processed_folds', root_folder='cmu_mosei')
    return true_scores_all


def compute_true_labels(y_list, all_scores, predict_neutral_class, threshold_emo_pres, num_classes, ext_name):
    """
    Compute the true labels from a list of 2D arrays of shape (1, num_classes) (y_list)
    The results are all 3D arrays, true_classes_pres is a list of arrays of this type)
    true_scores_all: Presence scores provided by the dataset. Shape: (number of y elements, 6)
    true_scores_coa: Closest true score among [0, 1, 2, 3]. Shape: (number of y elements, 6)
    true_classes_pres: Emotions truly present (varying thresholds for presence score given by threshold_emo_pres).
    Shape: (number of y elements, num_classes)
    true_classes_dom: Emotions truly dominant (highest presence score). Shape: (number of y elements, num_classes)

    :param y_list: List of arrays of shape (1, num_classes)
    :param all_scores: list of available presence scores
    :param predict_neutral_class: Whether we predict the neutral class
    :param threshold_emo_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param num_classes: number of emotion classes (7 with neutral class, else 6)
    :param ext_name: extension name containing info on whether we predict sentiment/neutral class
    :return: true_scores_all, true_scores_coa, true_classes_pres, true_classes_dom
    """

    if not pickle_file_exists("true_scores_all" + ext_name, pickle_folder='processed_folds', root_folder='cmu_mosei'):
        
        # Create 2D array from list of 2D true presence score arrays of shape (1, 6)
        true_scores_all = np.reshape(np.array(y_list), (-1, num_classes))
        save_with_pickle(true_scores_all, "true_scores_all" + ext_name, 'processed_folds', 'cmu_mosei')

        # Compute true presence scores: arrays of shape (4654, 6)
        # Possible values: [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]
        # Coarse-grained values: [0, 1, 2, 3]
        true_scores_coa = get_presence_score_from_finer_grained_val(true_scores_all, all_scores, num_classes, coarse=True)
        save_with_pickle(true_scores_coa, "true_scores_coa" + ext_name, 'processed_folds', 'cmu_mosei')

        # Compute true classes: binary arrays of shape (4654, 6 or 7)
        true_classes_pres = get_class_from_presence_score(true_scores_all, predict_neutral_class, threshold_emo_pres,
                                                          num_classes)
        true_classes_dom = get_class_from_presence_score(true_scores_all, predict_neutral_class, threshold_emo_pres,
                                                         num_classes, only_dominant=True)
        save_with_pickle(true_classes_pres, "true_classes_pres" + ext_name, 'processed_folds', 'cmu_mosei')
        save_with_pickle(true_classes_dom, "true_classes_dom" + ext_name, 'processed_folds', 'cmu_mosei')

    else:
        true_scores_all = load_from_pickle("true_scores_all" + ext_name, 'processed_folds', 'cmu_mosei')
        true_scores_coa = load_from_pickle("true_scores_coa" + ext_name, 'processed_folds', 'cmu_mosei')
        true_classes_pres = load_from_pickle("true_classes_pres" + ext_name, 'processed_folds', 'cmu_mosei')
        true_classes_dom = load_from_pickle("true_classes_dom" + ext_name, 'processed_folds', 'cmu_mosei')

    return true_scores_all, true_scores_coa, true_classes_pres, true_classes_dom


def compute_pred_labels(pred_raw, all_scores, predict_neutral_class, threshold_emo_pres, save_pred, model_id,
                        num_classes, model_folder):
    """
    Compute the prediction labels (all arrays of shape (test_size, 6), pred_classes_pres is a list of arrays of this type))
    pred_scores_all: Closest predicted score among [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]
    pred_scores_coa: Closest predicted score among [0, 1, 2, 3]
    pred_classes_pres: Emotions present predicted by the model (varying thresholds for presence score given by threshold_emo_pres)
    pred_classes_dom: Dominant emotions predicted by the model (highest presence score)

    :param pred_raw: array of shape (test_size, num_classes) predicting the presence score of the emotions
    :param all_scores: list of available presence scores
    :param predict_neutral_class: Whether we predict the neutral class
    :param threshold_emo_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param save_pred: Whether we save predictions with pickle
    :param model_id: Model id (int)
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param model_folder: The folder where the results will be saved
    :return: pred_scores_all, pred_scores_coa, pred_classes_pres, pred_classes_dom
    """

    # Get presence scores from raw predictions
    pred_scores_all = get_presence_score_from_finer_grained_val(pred_raw, all_scores, num_classes)
    pred_scores_coa = get_presence_score_from_finer_grained_val(pred_raw, all_scores, num_classes, coarse=True)

    # Compute predicted classes from presence scores (useful for classification metrics including confusion matrix)
    pred_classes_pres = get_class_from_presence_score(pred_scores_all, predict_neutral_class,
                                                      threshold_emo_pres, num_classes)
    pred_classes_dom = get_class_from_presence_score(pred_scores_all, predict_neutral_class,
                                                     threshold_emo_pres, num_classes, only_dominant=True)

    if save_pred:
        save_with_pickle(pred_scores_all, "pred_scores_all_{}".format(model_id),
                         pickle_folder="predictions", root_folder=model_folder)
        save_with_pickle(pred_scores_coa, "pred_scores_coa_{}".format(model_id),
                         pickle_folder="predictions", root_folder=model_folder)
        save_with_pickle(pred_classes_pres, "pred_classes_pres_{}".format(model_id),
                         pickle_folder="predictions", root_folder=model_folder)
        save_with_pickle(pred_classes_dom, "pred_classes_dom_{}".format(model_id),
                         pickle_folder="predictions", root_folder=model_folder)

    return pred_scores_all, pred_scores_coa, pred_classes_pres, pred_classes_dom

