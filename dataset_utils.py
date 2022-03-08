import os
import re
import pickle
import numpy as np
from mmsdk import mmdatasdk
from sklearn.utils import shuffle


# COLLAPSE FUNCTIONS FOR DATA ALIGNMENT

def avg_collapse_function(intervals: np.array, features: np.array) -> np.array:
    """
    Collapse function using average for visual and vocal modality alignment to textual modality.
    It only averages modality when it is possible (not the case for text modality).
    """
    try:
        return np.average(features, axis=0)
    except:
        return features


# CMU-MOSEI DATA DOWNLOAD FUNCTIONS

def download_dataset(dataset_folder, pickle_name, pickle_folder, align_to_text, append_label_to_data):
    """
    Download CMU-MOSEI dataset using the SDK and perform data alignment (if desired).

    :param dataset_folder: name of the folder where the CMU-MOSEI mmdataset will be downloaded
    :param pickle_name: name of the pickle object that will contain the CMU-MOSEI mmdataset
    :param pickle_folder: name of the folder where to save the pickle object
    :param align_to_text: whether we want data to align to the textual modality
    :param append_label_to_data: whether we want data to align to the labels
    """

    pickle_name += ".pkl"
    pickle_path = os.path.join(pickle_folder, pickle_name)

    if not os.path.isdir(pickle_folder):
        print("{} folder does not exist.".format(pickle_folder))

    elif os.path.exists(pickle_path):
        print("{} already exists in {} folder. Please change the pickle name.".format(pickle_name, pickle_folder))

    else:
        if os.path.exists('cmu_mosei'):
            # Retrieve data from cmu_mosei folder
            cmu_mosei = mmdatasdk.mmdataset(dataset_folder)
        else:
            # Download data and add them to cmu_mosei folder
            cmu_mosei = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.highlevel, dataset_folder)

        if align_to_text:
            cmu_mosei.align('glove_vectors', collapse_functions=[avg_collapse_function])

        if append_label_to_data:
            cmu_mosei.add_computational_sequences(mmdatasdk.cmu_mosei.labels, dataset_folder)
            cmu_mosei.align('All Labels')

        # Save cmu_mosei mmdataset with pickle
        with open(pickle_path, 'wb') as fw:
            pickle.dump(cmu_mosei, fw)


def load_dataset_from_pickle(pickle_name, pickle_folder):
    """
    Load CMU-MOSEI data from pickle file.

    :param pickle_name: name of the pickle object that contains the CMU-MOSEI mmdataset
    :param pickle_folder: name of the folder where to save the pickle object
    :return: cmu_mosei: mmdataset object containing all data (aligned or not)
    """

    pickle_name += ".pkl"
    pickle_path = os.path.join(pickle_folder, pickle_name)

    with open(pickle_path, 'rb') as fr:
        cmu_mosei = pickle.load(fr)

    return cmu_mosei


# CMU-MOSEI DATA SPLIT INTO TRAINING, VALIDATION AND TEST SETS

def get_fold_ids():
    """
    Get CMU-MOSEI standard fold ids for training, validation, and test

    :return: 3 lists of ids for training, validation, and test sets respectively
    """

    train_ids = mmdatasdk.cmu_mosei.standard_folds.standard_train_fold
    valid_ids = mmdatasdk.cmu_mosei.standard_folds.standard_valid_fold
    test_ids = mmdatasdk.cmu_mosei.standard_folds.standard_test_fold

    return train_ids, valid_ids, test_ids


def custom_split(train, valid):
    """
    Create three sets (training, validation and test) based on two sets (training and validation).
    Final training set:   85% of original training data
    Final validation set: 10% of original training data and 50% of validation data
    Final test set:        5% of original training data and 50% of validation data

    :param train: list of ids for training set
    :param valid: list of ids for validation set
    :return: 3 lists of ids for training, validation, and test sets respectively
    """
    total = len(valid)
    half = total // 2
    valid_ids_list = valid[:half]
    test_ids_list = valid[half + 1:]
    # 5 % of training into test data
    five_p = int(len(train) * 0.05)
    train_ids_list = train[:-five_p]
    test_ids_list = test_ids_list + train[-five_p:]
    # 10% of leftover training into valid data
    ten_p = int(len(train_ids_list) * 0.1)
    train_ids_list = train_ids_list[:-ten_p]
    valid_ids_list = valid_ids_list + train_ids_list[-ten_p:]
    return train_ids_list, valid_ids_list, test_ids_list


def split_dataset(dataset, train_ids, valid_ids, test_ids, image_feature):
    """
    Create 3 lists of arrays for training, validation and test sets.
    (Followed tutorial https://github.com/Justin1904/CMU-MultimodalSDK-Tutorials/blob/master/tutorial_interactive.ipynb)

    :param dataset: CMU-MOSEI mmdataset
    :param train_ids: list of training ids
    :param valid_ids: list of validation ids
    :param test_ids: list of test ids
    :param image_feature: image feature type (either FACET 4.2 or OpenFace 2)
    :return: 3 lists of training, validation and test sets containing lists of structure
            [text_seg, image_seg, audio_seg, label_seg, seg_id]
    """

    # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
    EPS = 0

    # Placeholders for training, validation, and test sets
    train = []
    valid = []
    test = []

    image_feature_id = 'FACET 4.2' if image_feature == 'facet' else 'OpenFace_2'
    image_dataset = dataset[image_feature_id]
    audio_dataset = dataset['COVAREP']
    text_dataset = dataset['glove_vectors']
    label_dataset = dataset['All Labels']

    pattern = re.compile('(.*)\[.*\]')
    num_drop = 0  # counter to get the number of data points that doesn't go through the checking process
    num_no_split = 0  # counter for those that doesn't belong to any split

    for seg_id in dataset['All Labels'].keys():

        # Get video id and its image, audio, text and label features
        vid = re.search(pattern, seg_id).group(1)

        if not (seg_id in image_dataset.keys() and seg_id in audio_dataset.keys() and seg_id in text_dataset.keys()):
            print("Encountered datapoint {} that does not appear in image, audio, or text dataset.".format(seg_id))
            num_drop += 1
            continue

        image_seg = image_dataset[seg_id]['features']
        audio_seg = audio_dataset[seg_id]['features']
        text_seg = text_dataset[seg_id]['features']
        label_seg = label_dataset[seg_id]['features']

        if not image_seg.shape[0] == audio_seg.shape[0] == text_seg.shape[0]:
            print("Encountered datapoint {} with image shape {}, audio shape {} and text shape {}."
                  .format(seg_id, image_seg.shape, audio_seg.shape, text_seg.shape))
            num_drop += 1
            continue

        # Remove nan values
        image_seg = np.nan_to_num(image_seg)
        audio_seg = np.nan_to_num(audio_seg)
        label_seg = np.nan_to_num(label_seg)

        # z-normalization per instance and remove nan/infs
        image_seg = np.nan_to_num(
            (image_seg - image_seg.mean(0, keepdims=True)) / (EPS + np.std(image_seg, axis=0, keepdims=True)))
        audio_seg = np.nan_to_num(
            (audio_seg - audio_seg.mean(0, keepdims=True)) / (EPS + np.std(audio_seg, axis=0, keepdims=True)))

        if vid in train_ids:
            train.append([text_seg, image_seg, audio_seg, label_seg, seg_id])
        elif vid in valid_ids:
            valid.append([text_seg, image_seg, audio_seg, label_seg, seg_id])
        elif vid in test_ids:
            test.append([text_seg, image_seg, audio_seg, label_seg, seg_id])
        else:
            # print("Encountered video {} that does not belong to any split.".format(vid))
            num_no_split += 1

    print("----------------------------------------------------")
    print("Split dataset complete!")
    total_data_split = len(train) + len(valid) + len(test)
    total_data = total_data_split + num_drop + num_no_split
    print("Total number of {} datapoints have been dropped ({:.2f}% of total data)."
          .format(num_drop, 100 * (num_drop / total_data)))
    print("Total number of {} datapoints do not belong to any split ({:.2f}% of total data)."
          .format(num_no_split, 100 * (num_no_split / total_data)))
    print("Number of training datapoints: {} ({:.2f}%)".format(len(train), 100 * (len(train) / total_data_split)))
    print("Number of validation datapoints: {} ({:.2f}%)".format(len(valid), 100 * (len(valid) / total_data_split)))
    print("Number of test datapoints: {} ({:.2f}%)".format(len(test), 100 * (len(test) / total_data_split)))

    return train, valid, test


def yield_datapoints(datapoint_list, random_state):
    """

    :param datapoint_list: list of datapoints containing lists of [text_seg, image_seg, audio_seg, label_seg, seg_id]
    :param random_state: random state for data shuffle
    :return: yields 1 array of shape (3, num_datapoints), 1 label and 1 segment id iteratively
    """
    datapoint_array = np.array(datapoint_list)
    X = datapoint_array[:, :3]
    y = datapoint_array[:, 3]
    seg_id = datapoint_array[:, 4]
    num_data = datapoint_array.shape[0]

    X_s, y_s, seg_id_s = shuffle(X, y, seg_id, random_state=random_state)

    for i in range(num_data):

        yield X_s[i], y_s[i], seg_id_s[i]
