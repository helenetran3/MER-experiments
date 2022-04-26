from src.pickle_functions import pickle_file_exists, load_from_pickle, create_extension_name
from src.dataset_utils import get_dataset_from_sdk, get_fold_ids, split_dataset, update_folds
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from src.csv_functions import save_model_param_in_csv_file, save_results_in_csv_file
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
                    help="Name of the model currently tested.")
parser.add_argument('-l', '--num_layers', type=int, choices=range(1, 4),
                    help="Number of bidirectional layers. Values between 1 and 3.")
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
parser.add_argument('-lf', '--loss_function', type=str,
                    help="Loss function")
parser.add_argument('-lr', '--learning_rate', type=float,
                    help="Learning rate")
parser.add_argument('-v', '--val_metric', type=str, choices=['loss', 'acc'],
                    help="Metric to monitor for validation set. Values: loss or acc.")
parser.add_argument('-nc', '--predict_neutral_class', action='store_true',
                    help="Predict neutral class.")
parser.add_argument('-emo', '--predict_sentiment', action='store_true',
                    help="Predict sentiment in addition to emotions.")
parser.add_argument('-tp', '--threshold_emo_present', type=float, nargs='+',
                    help="Threshold at which emotions are considered to be present. Values must be between 0 and 3. "
                         "Note that setting thresholds greater than 0 might lead to no positive true and predicted "
                         "classes and skew classification metrics (F1, precision, recall).")
parser.add_argument('-rd', '--round_decimals', type=int,
                    help="Number of decimals to be rounded for metrics.")
args = parser.parse_args()


def main():

    # Get data for training, validation and test sets from pickle (split provided by the SDK)
    if pickle_file_exists(args.pickle_name_fold + "_train", root_folder="cmu_mosei"):
        train_list = load_from_pickle(args.pickle_name_fold + "_train", root_folder="cmu_mosei")
        valid_list = load_from_pickle(args.pickle_name_fold + "_valid", root_folder="cmu_mosei")
        test_list = load_from_pickle(args.pickle_name_fold + "_test", root_folder="cmu_mosei")

    else:
        # Load CMU-MOSEI dataset
        dataset = get_dataset_from_sdk(args.pickle_name_dataset, args.align_to_text, args.append_label_to_data)

        # Get data for training, validation and test sets (split provided by the SDK)
        train_ids, valid_ids, test_ids = get_fold_ids(args.with_custom_split)
        train_list, valid_list, test_list = split_dataset(dataset,
                                                          train_ids, valid_ids, test_ids,
                                                          args.image_feature, args.pickle_name_fold)

    # Update labels for each fold, depending on predict_sentiment and predict_neutral_class values
    train_list, valid_list, test_list = update_folds(train_list, valid_list, test_list,
                                                     args.pickle_name_fold, args.predict_sentiment,
                                                     args.predict_neutral_class)

    # Model training
    train_model(train_list, valid_list, test_list,
                args.batch_size, args.num_epochs, args.fixed_num_steps, args.num_layers, args.num_nodes,
                args.dropout_rate, args.final_activ, args.learning_rate, args.loss_function, args.val_metric,
                args.patience, args.model_name, args.predict_neutral_class)

    extension_name = create_extension_name(args.predict_sentiment, args.predict_neutral_class)

    # Model evaluation
    loss_function_val, metrics_regression, metrics_score_coa, metrics_presence, metrics_dominant = \
        evaluate_model(test_list, args.batch_size, args.fixed_num_steps, args.num_layers, args.num_nodes,
                       args.dropout_rate, args.loss_function, args.model_name, args.predict_neutral_class,
                       args.predict_sentiment, args.threshold_emo_present, args.round_decimals, extension_name)

    # Save model parameters
    model_archi_param = [args.model_name, args.num_layers, args.num_nodes, args.dropout_rate, args.final_activ]
    model_archi_header = ['model_name', 'num_layers', 'num_nodes', 'dropout_rate', 'final_activ']
    model_id = save_model_param_in_csv_file(model_archi_param, model_archi_header, args.num_epochs, args.patience,
                                            args.batch_size, args.fixed_num_steps, args.loss_function,
                                            args.learning_rate, args.val_metric, args.predict_neutral_class,
                                            args.model_name)

    # Save metrics in csv file
    save_results_in_csv_file(args.model_name, model_id, args.loss_function, loss_function_val,
                             metrics_regression, metrics_score_coa, metrics_presence, metrics_dominant,
                             args.predict_neutral_class, args.threshold_emo_present, extension_name)


if __name__ == "__main__":
    main()
