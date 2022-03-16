from dataset_utils import download_dataset, load_dataset_from_pickle, get_fold_ids, perform_custom_split, split_dataset
from dataset_utils import get_dataset
import argparse
import os

parser = argparse.ArgumentParser(description="Emotion Recognition using CMU-MOSEI database. "
                                             "Related paper: "
                                             "Williams, J., Kleinegesse, S., Comanescu, R., & Radu, O. (2018, July). "
                                             "Recognizing Emotions in Video Using Multimodal DNN Feature Fusion. In "
                                             "Proceedings of Grand Challenge and Workshop on Human Multimodal "
                                             "Language (Challenge-HML) (pp. 11-19).")
parser.add_argument('-df', '--dataset_folder', type=str, default="cmu_mosei/",
                    help="Name of the folder where the CMU-MOSEI mmdataset will be downloaded (default: cmu_mosei/)")
parser.add_argument('-pn', '--pickle_name', type=str, default="cmu_mosei",
                    help="Name of the pickle object that will contain the CMU-MOSEI mmdataset (default: "
                         "cmu_mosei_aligned)")
parser.add_argument('-pf', '--pickle_folder', type=str, default="cmu_mosei/pickle_files/",
                    help="Name of the folder where to save the pickle object that contain the CMU-MOSEI mmdataset "
                         "(default: cmu_mosei/pickle_files/)")
parser.add_argument('-t', '--align_to_text', type=int, choices=range(0, 2), default=1,
                    help="Whether we want data to align to the textual modality. 1 for True (default) and 0 for False")
parser.add_argument('-l', '--append_label_to_data', type=int, choices=range(0, 2), default=1,
                    help="Whether we want data to append annotations to the dataset. 1 for True (default) and 0 for "
                         "False")
parser.add_argument('-c', '--with_custom_split', type=int, choices=range(0, 2), default=0,
                    help="Whether we want to perform custom split on training and validation sets (for more details, "
                         "cf. paper). 1 for True and 0 for False (default)")
parser.add_argument('-v', '--val_metric', type=str, choices=['loss', 'acc'], default='loss',
                    help="Metric to monitor for validation set. Values: loss (default) or acc.")
parser.add_argument('-f', '--image_feature', type=str, choices=['facet', 'openface'], default='facet',
                    help="Image features. Values: facet (default) or openface.")
parser.add_argument('-b', '--batch_size', type=str,
                    help="Batch size")
parser.add_argument('-s', '--fixed_num_steps', type=str,
                    help="Number of steps to fix for all sequences. Set to 0 if you want to keep the original number "
                         "of steps.")
args = parser.parse_args()


def main():
    pickle_path = os.path.join(args.pickle_folder, args.pickle_name + ".pkl")

    # Download CMU-MOSEI dataset using SDK and save with pickle
    if not os.path.exists(pickle_path):
        download_dataset(dataset_folder=args.dataset_folder,
                         pickle_name=args.pickle_name,
                         pickle_folder=args.pickle_folder,
                         align_to_text=args.align_to_text,
                         append_label_to_data=args.append_label_to_data)

    # Get CMU-MOSEI mmdataset object from pickle
    dataset = load_dataset_from_pickle(args.pickle_name, pickle_folder=args.pickle_folder)
    print("CMU-MOSEI dataset loaded")
    print("The existing computational sequences in dataset are: {}".format(list(dataset.keys())))

    # Get standard train, valid and test folds
    train_ids, valid_ids, test_ids = get_fold_ids()
    if args.with_custom_split:
        train_ids, valid_ids, test_ids = perform_custom_split(train_ids, valid_ids, test_ids)

    x_train, x_valid, x_test, y_train, y_valid, y_test, seg_train, seg_valid, seg_test = split_dataset(dataset, train_ids, valid_ids, test_ids, args.image_feature)

    # Create TensorFlow datasets for model training
    with_fixed_length = (args.fixed_num_steps > 0)
    train_dataset = get_dataset(x_train, y_train, seg_train, args.batch_size, with_fixed_length, args.fixed_num_steps)
    valid_dataset = get_dataset(x_valid, y_valid, seg_valid, args.batch_size, with_fixed_length, args.fixed_num_steps)
    test_dataset = get_dataset(x_test, y_test, seg_test, args.batch_size, with_fixed_length, args.fixed_num_steps)

if __name__ == "__main__":
    main()
