from dataset_utils import download_dataset, load_dataset_from_pickle, get_fold_ids, split_dataset
from model_training import train_model
import argparse
import os


parser = argparse.ArgumentParser(description="Emotion Recognition using CMU-MOSEI database. "
                                             "Related paper: "
                                             "Williams, J., Kleinegesse, S., Comanescu, R., & Radu, O. (2018, July). "
                                             "Recognizing Emotions in Video Using Multimodal DNN Feature Fusion. In "
                                             "Proceedings of Grand Challenge and Workshop on Human Multimodal "
                                             "Language (Challenge-HML) (pp. 11-19).")
parser.add_argument('-df', '--dataset_folder', type=str,
                    help="Name of the folder where the CMU-MOSEI mmdataset will be downloaded.")
parser.add_argument('-pnd', '--pickle_name_dataset', type=str,
                    help="Name of the pickle object that will contain the CMU-MOSEI mmdataset.")
parser.add_argument('-pnf', '--pickle_name_fold', type=str,
                    help="Name of the pickle object that will contain the training, validation and test folds.")
parser.add_argument('-pf', '--pickle_folder', type=str,
                    help="Name of the folder where to save the pickle object that contain the CMU-MOSEI mmdataset.")
parser.add_argument('-t', '--align_to_text', action='store_true',
                    help="Data will be aligned to the textual modality.")
parser.add_argument('-al', '--append_label_to_data', action='store_true',
                    help="Append annotations to the dataset.")
parser.add_argument('-c', '--with_custom_split', action='store_true',
                    help="Perform custom split on training and validation sets (for more details, cf. paper).")
parser.add_argument('-v', '--val_metric', type=str, choices=['loss', 'acc'],
                    help="Metric to monitor for validation set. Values: loss or acc.")
parser.add_argument('-f', '--image_feature', type=str, choices=['facet', 'openface'],
                    help="Image features. Values: facet or openface.")
parser.add_argument('-b', '--batch_size', type=int,
                    help="Batch size")
parser.add_argument('-s', '--fixed_num_steps', type=int,
                    help="Number of steps to fix for all sequences. Set to 0 if you want to keep the original number "
                         "of steps.")
parser.add_argument('-l', '--num_layers', type=int, choices=range(1, 4),
                    help="Number of bidirectional layers. Values between 1 and 3.")
parser.add_argument('-n', '--num_nodes', type=int,
                    help="Number of nodes in the penultimate dense layer.")
parser.add_argument('-d', '--dropout_rate', type=float,
                    help="Dropout rate")
parser.add_argument('-a', '--final_activ', type=str,
                    help="Activation function of the final layer.")
parser.add_argument('-md', '--model_dir', type=str,
                    help="Name of the directory where the models will be saved.")
parser.add_argument('-mn', '--model_name', type=str,
                    help="Name of the model to be saved.")
parser.add_argument('-e', '--num_epochs', type=int,
                    help="Maximum number of epochs")
parser.add_argument('-p', '--patience', type=int,
                    help="Number of epochs with no improvement after which the training will be stopped.")
parser.add_argument('-lr', '--learning_rate', type=float,
                    help="Learning rate")
parser.add_argument('-lf', '--loss_function', type=str,
                    help="Loss function")
args = parser.parse_args()


def main():
    pickle_dataset_path = os.path.join(args.pickle_folder, args.pickle_name_dataset + ".pkl")

    # Download CMU-MOSEI dataset using SDK and save with pickle
    if not os.path.exists(pickle_dataset_path):
        download_dataset(args.dataset_folder, args.pickle_name_dataset, args.pickle_folder,
                         args.align_to_text, args.append_label_to_data)

    # Get CMU-MOSEI mmdataset object from pickle #TODO: No need to load the dataset if we save the lists of folds
    dataset = load_dataset_from_pickle(args.pickle_name_dataset, args.pickle_folder)

    # Get ids of standard train, valid and test folds (provided by the SDK)
    train_ids, valid_ids, test_ids = get_fold_ids(args.with_custom_split)
    # pickle_train = args.pickle_name_fold + "_train.pkl"
    # pickle_train_path = os.path.join(args.pickle_folder, pickle_train)
    # if not os.path.exists(pickle_train_path):  # TODO: Check whether the pickle files containing the folds exist
    x_train, x_valid, x_test, y_train, y_valid, y_test, seg_train, seg_valid, seg_test = split_dataset(dataset,
                                                                                                       train_ids,
                                                                                                       valid_ids,
                                                                                                       test_ids,
                                                                                                       args.image_feature,
                                                                                                       args.pickle_name_fold,
                                                                                                       args.pickle_folder)

    # Model training
    train_model(x_train, x_valid, x_test, y_train, y_valid, y_test, seg_train, seg_valid, seg_test,
                args.batch_size, args.num_epochs, args.fixed_num_steps, args.num_layers,
                args.num_nodes, args.dropout_rate, args.final_activ, args.learning_rate, args.loss_function,
                args.val_metric, args.patience, args.model_dir, args.model_name)


if __name__ == "__main__":
    main()
