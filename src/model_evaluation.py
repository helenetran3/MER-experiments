import os

from tensorflow.keras.models import load_model
from src.pickle_functions import save_with_pickle
from src.dataset_utils import get_tf_dataset
from src.model_metrics import compute_loss_value, compute_multilabel_confusion_matrix, get_and_print_all_metrics
from src.label_processing import get_available_presence_scores, create_array_true_scores, compute_true_labels, compute_pred_labels


def model_prediction(model, test_dataset, num_test_samples, save_pred, model_id, model_folder):
    """
    Return the predictions of the model on test dataset.

    :param model: The model to evaluate
    :param test_dataset: The TensorFlow dataset for test set
    :param num_test_samples: Number of test samples
    :param save_pred: Whether we save predictions with pickle
    :param model_id: Model id (int)
    :param model_folder: The folder where the result will be saved
    :return: pred_raw: array giving the predictions of the model on test dataset.
    """

    print("\n\n================================= Model Prediction ===========================================\n")
    pred_raw = model.predict(test_dataset, verbose=1, steps=num_test_samples)  # (4654, num_classes)

    if save_pred:
        save_with_pickle(pred_raw, "pred_raw_{}".format(model_id), "predictions", model_folder)

    return pred_raw


def evaluate_model(test_list, batch_size, fixed_num_steps, loss_function,
                   model_name, model_id, predict_neutral_class, threshold_emo_pres, round_decimals,
                   ext_name, save_pred, save_confusion_matrix, display_fig):
    """
    Evaluate the performance of the best model.

    :param test_list: [x_test, y_test, seg_test]
    :param batch_size: Batch size for training
    :param fixed_num_steps: Fixed size for all the sequences (if we keep the original size, this parameter is set to 0)
    :param loss_function: Loss function
    :param model_name: Name of the model currently tested
    :param model_id: Model id (int)
    :param predict_neutral_class: Whether we predict the neutral class
    :param threshold_emo_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
    :param round_decimals: Number of decimals to be rounded for metrics
    :param ext_name: extension name containing info on whether we predict neutral class
    :param save_pred: Whether we save predictions with pickle
    :param save_confusion_matrix: Whether we save confusion matrices with pickle
    :param display_fig: Whether we display figures
    """

    # Load best model
    model_folder = os.path.join('models_tested', model_name)
    model_path = os.path.join(model_folder, 'models', "model_{}.h5".format(model_id))
    model = load_model(model_path)

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
    true_scores_all = create_array_true_scores(y_test, num_classes, ext_name)
    all_scores = get_available_presence_scores(true_scores_all)
    true_scores_coa, true_classes_pres, true_classes_dom = \
        compute_true_labels(true_scores_all, all_scores, predict_neutral_class, threshold_emo_pres, num_classes, ext_name)

    # Predicted labels
    pred_raw = model_prediction(model, test_dataset, num_test_samples, save_pred, model_id, model_folder)
    pred_scores, pred_scores_coa, pred_classes_pres, pred_classes_dom = \
        compute_pred_labels(pred_raw, all_scores, predict_neutral_class, threshold_emo_pres, save_pred, model_id,
                            num_classes, model_folder)

    # Confusion matrix
    if save_confusion_matrix:
        compute_multilabel_confusion_matrix(true_classes_pres, pred_classes_pres, threshold_emo_pres, num_classes,
                                            model_id, model_folder, display_fig)

    # Model evaluation
    loss_function_val = compute_loss_value(model, test_dataset, loss_function, round_decimals)

    # Compute all metrics
    metrics_regression, metrics_score_coa, metrics_presence, metrics_dominant = \
        get_and_print_all_metrics(true_scores_all, true_scores_coa, true_classes_pres, true_classes_dom,
                                  pred_raw, pred_scores_coa, pred_classes_pres, pred_classes_dom,
                                  threshold_emo_pres, num_classes, predict_neutral_class, round_decimals)

    return loss_function_val, metrics_regression, metrics_score_coa, metrics_presence, metrics_dominant
