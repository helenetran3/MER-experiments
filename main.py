from dataset_utils import download_dataset, load_dataset_from_pickle, get_fold_ids
from model_training import train_model
import argparse
import os


parser = argparse.ArgumentParser(description="Emotion Recognition using CMU-MOSEI database. "
                                             "Related paper: "
                                             "Williams, J., Kleinegesse, S., Comanescu, R., & Radu, O. (2018, July). "
                                             "Recognizing Emotions in Video Using Multimodal DNN Feature Fusion. In "
                                             "Proceedings of Grand Challenge and Workshop on Human Multimodal "
                                             "Language (Challenge-HML) (pp. 11-19).")
parser.add_argument('-df', '--dataset_folder', type=str, default="cmu_mosei/",
                    help="Name of the folder where the CMU-MOSEI mmdataset will be downloaded (default: cmu_mosei/).")
parser.add_argument('-pn', '--pickle_name', type=str, default="cmu_mosei",
                    help="Name of the pickle object that will contain the CMU-MOSEI mmdataset (default: "
                         "cmu_mosei_aligned).")
parser.add_argument('-pf', '--pickle_folder', type=str, default="cmu_mosei/pickle_files/",
                    help="Name of the folder where to save the pickle object that contain the CMU-MOSEI mmdataset "
                         "(default: cmu_mosei/pickle_files/).")
parser.add_argument('-t', '--align_to_text', type=int, choices=range(0, 2), default=1,
                    help="Whether we want data to align to the textual modality. 1 for True (default) and 0 for False.")
parser.add_argument('-al', '--append_label_to_data', type=int, choices=range(0, 2), default=1,
                    help="Whether we want data to append annotations to the dataset. 1 for True (default) and 0 for "
                         "False.")
parser.add_argument('-c', '--with_custom_split', type=int, choices=range(0, 2), default=0,
                    help="Whether we want to perform custom split on training and validation sets (for more details, "
                         "cf. paper). 1 for True and 0 for False (default).")
parser.add_argument('-v', '--val_metric', type=str, choices=['loss', 'acc'], default='loss',
                    help="Metric to monitor for validation set. Values: loss (default) or acc.")
parser.add_argument('-f', '--image_feature', type=str, choices=['facet', 'openface'], default='facet',
                    help="Image features. Values: facet (default) or openface.")
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
parser.add_argument('-a', '--final_activ', type=str, default='linear',
                    help="Activation function of the final layer (default: linear).")
args = parser.parse_args()


def main():
    if args.fixed_num_steps is None:
        print("fixed_num_steps (-s) not set, please check again.")
        quit()
    if args.num_nodes is None:
        print("num_nodes (-n) not set, please check again.")
        quit()
    if args.num_layers is None:
        print("num_layers (-l) not set, please check again.")
        quit()
    if args.batch_size is None:
        print("batch_size (-b) not set, please check again.")
        quit()

    pickle_path = os.path.join(args.pickle_folder, args.pickle_name + ".pkl")

    # Download CMU-MOSEI dataset using SDK and save with pickle
    if not os.path.exists(pickle_path):
        download_dataset(args.dataset_folder, args.pickle_name, args.pickle_folder,
                         args.align_to_text, args.append_label_to_data)

    # Get CMU-MOSEI mmdataset object from pickle
    dataset = load_dataset_from_pickle(args.pickle_name, args.pickle_folder)

    # Get ids of standard train, valid and test folds (provided by the SDK)
    train_ids, valid_ids, test_ids = get_fold_ids(args.with_custom_split)

    # Model training
    train_model(dataset, train_ids, valid_ids, test_ids,
                args.batch_size, args.fixed_num_steps, args.image_feature,
                args.num_layers, args.num_nodes, args.dropout_rate, args.final_activ)


if __name__ == "__main__":
    main()
