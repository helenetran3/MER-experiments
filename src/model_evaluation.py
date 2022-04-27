import numpy as np

from tensorflow.keras.models import load_model
from src.pickle_functions import *
from src.dataset_utils import get_tf_dataset
from src.model_metrics import get_and_print_all_metrics, compute_multilabel_confusion_matrix, compute_loss_value

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer


# COMPUTE TRUE/PREDICTED LABEL ARRAYS DERIVED FROM PRESENCE SCORES: ALL PRESENCE SCORES, COARSE PRESENCE SCORES,
# BINARY CLASSIFICATION FOR EACH EMOTION, CLASSIFICATION OF DOMINANT EMOTIONS


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


def create_array_true_scores(y_test, num_classes, extension_name):
    """
    Return the true scores given by the database.
    Possible values: [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]

    :param y_test: List of arrays of shape (1, num_classes)
    :param num_classes: Number of classes
    :param extension_name: extension name containing info on whether we predict sentiment/neutral class
    :return: true_scores_all
    """

    if not pickle_file_exists("true_scores_all" + extension_name, root_folder='cmu_mosei'):
        true_scores_all = np.reshape(np.array(y_test), (-1, num_classes))
        save_with_pickle(true_scores_all, "true_scores_all" + extension_name, root_folder='cmu_mosei')
    else:
        true_scores_all = load_from_pickle("true_scores_all" + extension_name, root_folder='cmu_mosei')
    return true_scores_all


def compute_true_labels(true_scores_all, predict_neutral_class, threshold_emo_pres, num_classes, extension_name):
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
    :param extension_name: extension name containing info on whether we predict sentiment/neutral class
    :return: true_scores_coa, true_classes_pres, true_classes_dom
    """

    if not pickle_file_exists("true_scores_coarse" + extension_name, root_folder='cmu_mosei'):

        # Compute true presence scores: arrays of shape (4654, 7)
        # Possible values: [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]
        # Coarse-grained values: [0, 1, 2, 3]
        true_scores_coa = get_presence_score_from_finer_grained_val(true_scores_all, true_scores_all, num_classes, coarse=True)
        save_with_pickle(true_scores_coa, "true_scores_coarse" + extension_name, root_folder='cmu_mosei')

        # Compute true classes: binary arrays of shape (4654, 6 or 7)
        true_classes_pres = get_class_from_presence_score(true_scores_all, predict_neutral_class, threshold_emo_pres,
                                                          num_classes)
        true_classes_dom = get_class_from_presence_score(true_scores_all, predict_neutral_class, threshold_emo_pres,
                                                         num_classes, only_dominant=True)
        save_with_pickle(true_classes_pres, "true_classes_pres" + extension_name, root_folder='cmu_mosei')
        save_with_pickle(true_classes_dom, "true_classes_dom" + extension_name, root_folder='cmu_mosei')

    else:
        true_scores_coa = load_from_pickle("true_scores_coarse" + extension_name, root_folder='cmu_mosei')
        true_classes_pres = load_from_pickle("true_classes_pres" + extension_name, root_folder='cmu_mosei')
        true_classes_dom = load_from_pickle("true_classes_dom" + extension_name, root_folder='cmu_mosei')

    return true_scores_coa, true_classes_pres, true_classes_dom


def compute_pred_labels(pred_raw, true_scores_all, predict_neutral_class, threshold_emo_pres, model_id,
                        num_classes, model_folder):
    """
    Compute the prediction labels (all arrays of shape (test_size, 6), pred_classes_pres is a list of arrays of this type))
    pred_scores_all: Closest predicted score among [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]
    pred_scores_coa: Closest predicted score among [0, 1, 2, 3]
    pred_classes_pres: Emotions present predicted by the model (varying thresholds for presence score given by threshold_emo_pres)
    pred_classes_dom: Dominant emotions predicted by the model (highest presence score)

    :param pred_raw: array of shape (test_size, num_classes) predicting the presence score of the emotions
    :param true_scores_all: array of shape (test size, num_classes) giving the true scores for the emotions (given by
    the database)
    :param predict_neutral_class: Whether we predict the neutral class
    :param threshold_emo_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param model_id: Model id (int)
    :param num_classes: number of classes (7 with neutral class, else 6)
    :param model_folder: The folder where the results will be saved
    :return: pred_scores_all, pred_scores_coa, pred_classes_pres, pred_classes_dom
    """

    pred_sc_save_name = "pred_scores_all_{}".format(model_id)
    pred_sc_coa_save_name = "pred_scores_coarse_{}".format(model_id)
    pred_cl_pres_save_name = "pred_classes_pres_{}".format(model_id)
    pred_cl_dom_save_name = "pred_classes_dom_{}".format(model_id)

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


def model_prediction(model, test_dataset, num_test_samples, model_id, model_folder):
    """
    Return the predictions of the model on test dataset.

    :param model: The model to evaluate
    :param test_dataset: The TensorFlow dataset for test set
    :param num_test_samples: Number of test samples
    :param model_id: Model id (int)
    :param model_folder: The folder where the result will be saved
    :return: pred_raw: array giving the predictions of the model on test dataset.
    """

    pred_raw_save_name = "pred_raw_{}".format(model_id)

    # Get raw score predictions from the model
    if not pickle_file_exists(pred_raw_save_name, root_folder=model_folder):  # perform prediction (for debugging)
        print("\n\n================================= Model Prediction ===========================================\n")
        pred_raw = model.predict(test_dataset, verbose=1, steps=num_test_samples)  # (4654, num_classes)
        save_with_pickle(pred_raw, pred_raw_save_name, root_folder=model_folder)
    else:
        pred_raw = load_from_pickle(pred_raw_save_name, root_folder=model_folder)

    return pred_raw


def evaluate_model(test_list, batch_size, fixed_num_steps, loss_function,
                   model_name, model_id, predict_neutral_class, threshold_emo_pres, round_decimals,
                   extension_name):
    """
    Evaluate the performance of the best model.

    :param test_list: [x_test, y_test, seg_test]
    :param batch_size: Batch size for training
    :param fixed_num_steps: Fixed size for all the sequences (if we keep the original size, this parameter is set to 0)
    # :param num_layers: Number of bidirectional layers for the model
    # :param num_nodes: Number of nodes for the penultimate dense layer
    # :param dropout_rate: Dropout rate before each dense layer
    :param loss_function: Loss function
    :param model_name: Name of the model currently tested
    :param model_id: Model id (int)
    :param predict_neutral_class: Whether we predict the neutral class
    :param threshold_emo_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param round_decimals: Number of decimals to be rounded for metrics
    :param extension_name: extension name containing info on whether we predict sentiment/neutral class
    """

    # Load best model
    model_save_name = "model_{}.h5".format(model_id)
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
    true_scores_all = create_array_true_scores(y_test, num_classes, extension_name)
    true_scores_coa, true_classes_pres, true_classes_dom = \
        compute_true_labels(true_scores_all, predict_neutral_class, threshold_emo_pres, num_classes, extension_name)

    # Predicted labels
    pred_raw = model_prediction(model, test_dataset, num_test_samples, model_id, model_folder)
    pred_scores, pred_scores_coa, pred_classes_pres, pred_classes_dom = \
        compute_pred_labels(pred_raw, true_scores_all, predict_neutral_class, threshold_emo_pres, model_id,
                            num_classes, model_folder)

    # Confusion matrix
    compute_multilabel_confusion_matrix(true_classes_pres, pred_classes_pres, threshold_emo_pres, num_classes,
                                        model_id, model_folder)

    # Model evaluation
    loss_function_val = compute_loss_value(model, test_dataset, loss_function, round_decimals)

    # Compute all metrics
    metrics_regression, metrics_score_coa, metrics_presence, metrics_dominant = \
        get_and_print_all_metrics(true_scores_all, true_scores_coa, true_classes_pres, true_classes_dom,
                                  pred_raw, pred_scores_coa, pred_classes_pres, pred_classes_dom,
                                  threshold_emo_pres, num_classes, predict_neutral_class, round_decimals)

    return loss_function_val, metrics_regression, metrics_score_coa, metrics_presence, metrics_dominant
