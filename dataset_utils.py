import os
import pickle
import numpy as np
from mmsdk import mmdatasdk


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

def download_dataset(pickle_name, pickle_folder, align_to_text, append_label_to_data):
    """
    Download CMU-MOSEI dataset using the SDK and perform data alignment (if desired).

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
            cmu_mosei = mmdatasdk.mmdataset('cmu_mosei/')
        else:
            # Download data and add them to cmu_mosei folder
            cmu_mosei = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.highlevel, 'cmu_mosei/')

        if align_to_text:
            cmu_mosei.align('glove_vectors', collapse_functions=[avg_collapse_function])

        if append_label_to_data:
            cmu_mosei.add_computational_sequences(mmdatasdk.cmu_mosei.labels, 'cmu_mosei/')
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


# TODO: Create vectors of shape (dataset_size, max_len, feature_dim) for all sets
# def split_dataset(dataset, train_ids, valid_ids, test_ids, mode, max_len):
#
#     return x_train, x_valid, x_test, y_train, y_valid, y_test
