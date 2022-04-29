from src.pickle_functions import pickle_file_exists, load_from_pickle
from src.dataset_utils import get_dataset_from_sdk, get_fold_ids, split_dataset
from src.model_training import train_model, get_optimizer
from src.model_evaluation import evaluate_model
from src.csv_functions import save_model_param_in_csv_file, save_results_in_csv_file, get_header_and_data_model
import argparse

parser = argparse.ArgumentParser(description="SOTA Multimodal Emotion Recognition models using CMU-MOSEI database.")
parser.add_argument('-pnd', '--pickle_name_dataset', type=str,
                    help="Name of the pickle object that will contain the CMU-MOSEI mmdataset.")
parser.add_argument('-pnf', '--pickle_name_fold', type=str,
                    help="Name of the pickle object that will contain the training, validation and test folds.")
parser.add_argument('-t', '--align_to_text', action='store_true',
                    help="Data will be aligned to the textual modality.")
parser.add_argument('-al', '--append_label_to_data', action='store_true',
                    help="Append annotations to the dataset.")
parser.add_argument('-f', '--image_feature', type=str, choices=['facet', 'openface'],
                    help="Image features. Values: facet or openface.")
parser.add_argument('-c', '--with_custom_split', action='store_true',
                    help="Perform custom split on training and validation sets (for more details, cf. Williams et al. "
                         "(2018) paper).")
parser.add_argument('-mn', '--model_name', type=str,
                    help="Name of the model currently tested. Values: ef_williams.")
parser.add_argument('-l', '--num_layers', type=int,
                    help="Number of bidirectional layers.")
parser.add_argument('-n', '--num_nodes', type=int,
                    help="Number of nodes in the penultimate dense layer.")
parser.add_argument('-d', '--dropout_rate', type=float,
                    help="Dropout rate")
parser.add_argument('-a', '--final_activ', type=str,
                    help="Activation function of the final layer.")
parser.add_argument('-e', '--num_epochs', type=int,
                    help="Maximum number of epochs")
parser.add_argument('-p', '--patience', type=int,
                    help="Number of epochs with no improvement after which the training will be stopped.")
parser.add_argument('-b', '--batch_size', type=int,
                    help="Batch size")
parser.add_argument('-s', '--fixed_num_steps', type=int,
                    help="Number of steps to fix for all sequences. Set to 0 if you want to keep the original number "
                         "of steps.")
parser.add_argument('-opt', '--optimizer', type=str,
                    help="Optimizer for model training. Values: adam, sgd, adagrad, adadelta, rmsprop.")
parser.add_argument('-lf', '--loss_function', type=str,
                    help="Loss function")
parser.add_argument('-lr', '--learning_rate', type=float,
                    help="Learning rate")
parser.add_argument('-v', '--val_metric', type=str, choices=['loss', 'acc'],
                    help="Metric to monitor for validation set. Values: loss or acc.")
parser.add_argument('-nc', '--predict_neutral_class', action='store_true',
                    help="Predict neutral class.")
parser.add_argument('-tp', '--threshold_emo_present', type=float, nargs='+',
                    help="Threshold at which emotions are considered to be present. Values must be between 0 and 3. "
                         "Note that setting thresholds greater than 0 might lead to no positive true and predicted "
                         "classes and skew classification metrics (F1, precision, recall).")
parser.add_argument('-rd', '--round_decimals', type=int,
                    help="Number of decimals to be rounded for metrics.")
parser.add_argument('-sp', '--save_predictions', action='store_true',
                    help="Save predictions with pickle")
parser.add_argument('-scm', '--save_confusion_matrix', action='store_true',
                    help="Save confusion matrix with pickle")
parser.add_argument('-df', '--display_fig', action='store_true',
                    help="Whether we display the figures")
args = parser.parse_args()


def main():

    # Placed at the beginning in order to verify optimizer name quickly
    optimizer_tf = get_optimizer(args.optimizer)

    # Save model parameters and return model id
    model_header, model_data = get_header_and_data_model(args.model_name, args.num_layers, args.num_nodes,
                                                         args.dropout_rate, args.final_activ)
    model_id = save_model_param_in_csv_file(model_data, model_header, args.num_epochs, args.patience,
                                            args.batch_size, args.fixed_num_steps, args.optimizer, args.loss_function,
                                            args.learning_rate, args.val_metric, args.predict_neutral_class,
                                            args.model_name)

    # Get data for training, validation and test sets from pickle (split provided by the SDK)
    if pickle_file_exists(args.pickle_name_fold + "_train", "raw_folds", "cmu_mosei"):
        train_list = load_from_pickle(args.pickle_name_fold + "_train", "raw_folds", "cmu_mosei")
        valid_list = load_from_pickle(args.pickle_name_fold + "_valid", "raw_folds", "cmu_mosei")
        test_list = load_from_pickle(args.pickle_name_fold + "_test",  "raw_folds", "cmu_mosei")

    else:
        # Load CMU-MOSEI dataset
        dataset = get_dataset_from_sdk(args.pickle_name_dataset, args.align_to_text, args.append_label_to_data)

        # Get data for training, validation and test sets (split provided by the SDK)
        train_ids, valid_ids, test_ids = get_fold_ids(args.with_custom_split)
        train_list, valid_list, test_list = split_dataset(dataset,
                                                          train_ids, valid_ids, test_ids,
                                                          args.image_feature, args.pickle_name_fold)

    all_scores = [0, 0.16, 0.33, 0.5, 0.66, 1, 1.33, 1.66, 2, 2.33, 2.66, 3]  # TODO: get it from training fold

    # Model training
    train_model(train_list, valid_list, test_list,
                args.batch_size, args.num_epochs, args.fixed_num_steps, args.num_layers, args.num_nodes,
                args.dropout_rate, args.final_activ, args.learning_rate, optimizer_tf, args.optimizer,
                args.loss_function, args.val_metric, args.patience, args.model_name, args.predict_neutral_class,
                model_id, args.display_fig)

    # Model evaluation
    ext_name = "_n" if args.predict_neutral_class else ""
    loss_function_val, metrics_regression, metrics_score_coa, metrics_presence, metrics_dominant = \
        evaluate_model(test_list, args.batch_size, args.fixed_num_steps, args.loss_function, args.model_name, model_id,
                       all_scores, args.predict_neutral_class, args.threshold_emo_present, args.round_decimals, ext_name,
                       args.save_predictions, args.save_confusion_matrix, args.display_fig)

    # Save metrics in csv file
    save_results_in_csv_file(args.model_name, model_id, args.loss_function, loss_function_val,
                             metrics_regression, metrics_score_coa, metrics_presence, metrics_dominant,
                             args.predict_neutral_class, args.threshold_emo_present, ext_name)


if __name__ == "__main__":
    main()
