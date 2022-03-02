from dataset_utils import download_dataset, load_dataset_from_pickle, get_fold_ids, custom_split
import argparse
import os

parser = argparse.ArgumentParser(description="Emotion Recognition using CMU-MOSEI database. "
                                             "Related paper: "
                                             "Williams, J., Kleinegesse, S., Comanescu, R., & Radu, O. (2018, July). "
                                             "Recognizing Emotions in Video Using Multimodal DNN Feature Fusion. In "
                                             "Proceedings of Grand Challenge and Workshop on Human Multimodal "
                                             "Language (Challenge-HML) (pp. 11-19).")
parser.add_argument('-pickle_name', type=str, default="cmu_mosei_aligned",
                    help="Name of the pickle object that will contain the CMU-MOSEI mmdataset (default: "
                         "cmu_mosei_aligned)")
parser.add_argument('-pickle_folder', type=str, default="cmu_mosei/pickle_files/",
                    help="Name of the folder where to save the pickle object that contain the CMU-MOSEI mmdataset "
                         "(default: cmu_mosei/pickle_files/)")
parser.add_argument('-align_to_text', type=int, choices=range(0, 2), default=1,
                    help="Whether we want data to align to the textual modality. 1 for True (default) and 0 for False")
parser.add_argument('-append_label_to_data', type=int, choices=range(0, 2), default=1,
                    help="Whether we want data to append annotations to the dataset. 1 for True (default) and 0 for "
                         "False")
parser.add_argument('-with_custom_split', type=int, choices=range(0, 2), default=0,
                    help="Whether we want to perform custom split (cf. paper). 1 for True and 0 for False (default)")
parser.add_argument('-val_metric', type=str, choices=['loss', 'acc'], default='loss',
                    help="Metric to monitor for validation set. Values: loss (default) or acc.")
args = parser.parse_args()


def main():
    pickle_path = os.path.join(args.pickle_folder, args.pickle_name + ".pkl")

    # Download CMU-MOSEI dataset using SDK and save with pickle
    if not os.path.exists(pickle_path):
        download_dataset(args.pickle_name, pickle_folder=args.pickle_folder, align_to_text=args.align_to_text,
                         append_label_to_data=args.append_label_to_data)

    # Get CMU-MOSEI mmdataset object from pickle
    dataset = load_dataset_from_pickle(args.pickle_name, pickle_folder=args.pickle_folder)
    print("CMU-MOSEI dataset loaded")
    print("The existing computational sequences in dataset are: {}".format(list(dataset.keys())))

    # Get standard train, valid and test folds
    train_ids, valid_ids, test_ids = get_fold_ids()
    if args.with_custom_split:
        train_ids_cs, valid_ids_cs, test_ids_cs = custom_split(train_ids, valid_ids)
        train_ids, valid_ids, test_ids = train_ids_cs, valid_ids_cs, test_ids_cs + test_ids


if __name__ == "__main__":
    main()
