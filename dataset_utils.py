import os
import re
import pickle
import numpy as np
import tensorflow as tf
from mmsdk import mmdatasdk


# CMU-MOSEI DATA DOWNLOAD FUNCTIONS

def avg_collapse_function(intervals: np.array, features: np.array) -> np.array:
    """
    Collapse function for data alignment using average for visual and vocal modality alignment to textual modality.
    It only averages modality when it is possible (not the case for text modality).
    """
    try:
        return np.average(features, axis=0)
    except:
        return features


def download_dataset(dataset_folder, pickle_name_dataset, pickle_folder, align_to_text, append_label_to_data):
    """
    Download CMU-MOSEI dataset using the SDK and perform data alignment (if desired).

    :param dataset_folder: name of the folder where the CMU-MOSEI mmdataset will be downloaded
    :param pickle_name_dataset: name of the pickle object that will contain the CMU-MOSEI mmdataset
    :param pickle_folder: name of the folder where to save the pickle object
    :param align_to_text: whether we want data to align to the textual modality
    :param append_label_to_data: whether we want data to align to the labels
    """

    pickle_name_dataset += ".pkl"
    pickle_path = os.path.join(pickle_folder, pickle_name_dataset)

    if not os.path.isdir(pickle_folder):
        os.mkdir(pickle_folder)

    if os.path.exists(pickle_path):  # For safety. The function should be called only if the pickle obj does not exist
        print("{} already exists in {} folder. Please change the pickle name.".format(pickle_name_dataset, pickle_folder))

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


def pickle_file_exists(pickle_name, pickle_folder, dataset_else_fold):
    """
    Check whether a pickle file exists (mmdataset or list of folds)

    :param pickle_name: name of the pickle object
    :param pickle_folder: name of the folder where the pickle object is saved
    :param dataset_else_fold: True if we want to check whether CMU-MOSEI mmdataset exists, False for lists of training,
    validation and test folds
    :return: True if the pickle file exists, else False
    """

    pickle_name_to_check = pickle_name + ".pkl" if dataset_else_fold else pickle_name + "_train.pkl"
    pickle_path = os.path.join(pickle_folder, pickle_name_to_check)

    return os.path.exists(pickle_path)


def load_dataset_from_pickle(pickle_name, pickle_folder):
    """
    Load CMU-MOSEI dataset from pickle file.

    :param pickle_name: name of the pickle object that contains the CMU-MOSEI mmdataset
    :param pickle_folder: name of the folder where the pickle object is saved
    :return: cmu_mosei: mmdataset object containing all data (aligned or not)
    """

    pickle_name += ".pkl"
    pickle_path = os.path.join(pickle_folder, pickle_name)

    with open(pickle_path, 'rb') as fr:
        cmu_mosei = pickle.load(fr)

    print("CMU-MOSEI dataset loaded from pickle.")
    print("The existing computational sequences in dataset are: {}".format(list(cmu_mosei.keys())))

    return cmu_mosei


def get_dataset_from_sdk(dataset_folder, pickle_name_dataset, pickle_folder, align_to_text, append_label_to_data):
    """
    Return CMU-MOSEI mmdataset.

    :param dataset_folder: name of the folder where the CMU-MOSEI mmdataset will be downloaded
    :param pickle_name_dataset: name of the pickle object that will contain the CMU-MOSEI mmdataset
    :param pickle_folder: name of the folder where to save the pickle object
    :param align_to_text: whether we want data to align to the textual modality
    :param append_label_to_data: whether we want data to align to the labels
    :return: CMU-MOSEI mmdataset
    """

    if not pickle_file_exists(pickle_name_dataset, pickle_folder, dataset_else_fold=True):
        download_dataset(dataset_folder, pickle_name_dataset, pickle_folder, align_to_text, append_label_to_data)

    # Get CMU-MOSEI mmdataset object from pickle
    dataset = load_dataset_from_pickle(pickle_name_dataset, pickle_folder)

    return dataset


# CMU-MOSEI DATA SPLIT INTO TRAINING, VALIDATION AND TEST SETS

def perform_custom_split(train_ids, valid_ids, test_ids):
    """
    Perform custom split on training and validation sets.
    Final training set:   85% of original training data
    Final validation set: 10% of training data and 50% of validation data
    Final test set:       (5% of training data and 50% of validation data) + original test data

    :param train_ids: list of segment ids for training set
    :param valid_ids: list of segment ids for validation set
    :param test_ids: list of segment ids for test set
    :return: new train_ids, valid_ids, test_ids
    """

    def custom_split(train, valid):
        """
        Create three sets (training, validation and test) based on two sets (training and validation).
        Final training set:   85% of original training data
        Final validation set: 10% of training data and 50% of validation data
        Final test set:        5% of training data and 50% of validation data

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

    train_ids_cs, valid_ids_cs, test_ids_cs = custom_split(train_ids, valid_ids)
    new_train_ids, new_valid_ids, new_test_ids = train_ids_cs, valid_ids_cs, test_ids_cs + test_ids

    return new_train_ids, new_valid_ids, new_test_ids


def get_fold_ids(with_custom_split):
    """
    Get CMU-MOSEI standard fold ids for training, validation, and test

    :param with_custom_split: whether we perform custom_split proposed by the paper written by Williams et al. (ACL 2018)
    :return: 3 lists of ids for training, validation, and test sets respectively
    """

    train_ids = mmdatasdk.cmu_mosei.standard_folds.standard_train_fold
    valid_ids = mmdatasdk.cmu_mosei.standard_folds.standard_valid_fold
    test_ids = mmdatasdk.cmu_mosei.standard_folds.standard_test_fold

    if with_custom_split:
        train_ids, valid_ids, test_ids = perform_custom_split(train_ids, valid_ids, test_ids)

    return train_ids, valid_ids, test_ids


def split_dataset(dataset, train_ids, valid_ids, test_ids, image_feature, pickle_name_fold, pickle_folder):
    """
    For each training, validation and test sets, create three lists:
    - one for features (x): arrays of shape (number steps, number features) for text/image/audio features (concatenated
    in this order)
    - one for labels (y): arrays of shape (1, 7), for the 7 emotions
    - one for segment ids (seg): id of the segment described by (x, y). Example: 'zk2jTlAtvSU[1]'
    (Followed tutorial https://github.com/Justin1904/CMU-MultimodalSDK-Tutorials/blob/master/tutorial_interactive.ipynb)
    Note that this function performs **early fusion** (concatenation of low level text, image and audio features).

    :param dataset: CMU-MOSEI mmdataset
    :param train_ids: list of training ids
    :param valid_ids: list of validation ids
    :param test_ids: list of test ids
    :param image_feature: image feature type (either FACET 4.2 or OpenFace 2)
    :param pickle_name_fold: root name of the pickle object that contains the training, validation and test folds
    :param pickle_folder: name of the folder where the pickle object is saved
    :return: 3 lists train_list = [x_train, y_train, seg_train], valid_list = [x_valid, y_valid, seg_valid],
     test_list = [x_test, y_test, seg_test]
    """

    # a sentinel epsilon for safe division, without it we will replace illegal values with a constant
    EPS = 0

    # Placeholders for training, validation, and test sets
    x_train, y_train, seg_train = [], [], []
    x_valid, y_valid, seg_valid = [], [], []
    x_test, y_test, seg_test = [], [], []

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

        data_seg = np.concatenate((text_seg, image_seg, audio_seg), axis=1)  # early fusion

        if vid in train_ids:
            x_train.append(data_seg)
            y_train.append(label_seg)
            seg_train.append(seg_id)
        elif vid in valid_ids:
            x_valid.append(data_seg)
            y_valid.append(label_seg)
            seg_valid.append(seg_id)
        elif vid in test_ids:
            x_test.append(data_seg)
            y_test.append(label_seg)
            seg_test.append(seg_id)
        else:
            # print("Encountered video {} that does not belong to any split.".format(vid))
            num_no_split += 1

    train_size = len(y_train)
    valid_size = len(y_valid)
    test_size = len(y_test)

    print("----------------------------------------------------")
    print("Split dataset complete!")
    total_data_split = train_size + valid_size + test_size
    total_data = total_data_split + num_drop + num_no_split
    print("Total number of {} datapoints have been dropped ({:.2f}% of total data)."
          .format(num_drop, 100 * (num_drop / total_data)))
    print("Total number of {} datapoints do not belong to any split ({:.2f}% of total data)."
          .format(num_no_split, 100 * (num_no_split / total_data)))
    print("Number of training datapoints: {} ({:.2f}%)".format(train_size, 100 * (train_size / total_data_split)))
    print("Number of validation datapoints: {} ({:.2f}%)".format(valid_size, 100 * (valid_size / total_data_split)))
    print("Number of test datapoints: {} ({:.2f}%)".format(test_size, 100 * (test_size / total_data_split)))

    train_res = [x_train, y_train, seg_train]
    valid_res = [x_valid, y_valid, seg_valid]
    test_res = [x_test, y_test, seg_test]

    # Save lists with pickle
    if not os.path.isdir(pickle_folder):
        os.mkdir(pickle_folder)
    pickle_train = pickle_name_fold + "_train.pkl"
    pickle_valid = pickle_name_fold + "_valid.pkl"
    pickle_test = pickle_name_fold + "_test.pkl"
    pickle_train_path = os.path.join(pickle_folder, pickle_train)
    pickle_valid_path = os.path.join(pickle_folder, pickle_valid)
    pickle_test_path = os.path.join(pickle_folder, pickle_test)
    with open(pickle_train_path, 'wb') as fw_train:
        pickle.dump(train_res, fw_train)
    with open(pickle_valid_path, 'wb') as fw_valid:
        pickle.dump(valid_res, fw_valid)
    with open(pickle_test_path, 'wb') as fw_test:
        pickle.dump(test_res, fw_test)

    return train_res, valid_res, test_res


def load_folds_from_pickle(pickle_name, pickle_folder):
    """
    Load training, validation and test folds from pickle file.

    :param pickle_name: root name of the pickle object that contains the training, validation and test folds
    :param pickle_folder: name of the folder where the pickle object is saved
    :return: train_list, valid_list, test_list: 3 lists of [x, y, seg_id] for training, validation and test sets
    respectively.
    x is a list of arrays of shape (number steps, number features)
    y a list of arrays of shape (1, 7) for the 7 emotions
    seg_id a list of ids of the segment described by (x, y) (ex: 'zk2jTlAtvSU[1]')
    """

    pickle_train = pickle_name + "_train.pkl"
    pickle_valid = pickle_name + "_valid.pkl"
    pickle_test = pickle_name + "_test.pkl"
    pickle_train_path = os.path.join(pickle_folder, pickle_train)
    pickle_valid_path = os.path.join(pickle_folder, pickle_valid)
    pickle_test_path = os.path.join(pickle_folder, pickle_test)

    with open(pickle_train_path, 'rb') as fr_train:
        train_list = pickle.load(fr_train)
    with open(pickle_valid_path, 'rb') as fr_valid:
        valid_list = pickle.load(fr_valid)
    with open(pickle_test_path, 'rb') as fr_test:
        test_list = pickle.load(fr_test)

    print("Training, validation, and test data loaded from pickle.")

    return train_list, valid_list, test_list


def get_data_folds(dataset, with_custom_split, pickle_name_fold, pickle_folder, image_feature):
    """
    Get three lists for training, validation and test sets respectively. Each list contains:
    - a list of features (x): arrays of shape (number steps, number features) for text/image/audio features
    (concatenated in this order)
    - a list of labels (y): arrays of shape (1, 7), for the 7 emotions
    - a list of segment ids (seg): id of the segment described by (x, y). Example: 'zk2jTlAtvSU[1]'

    :param dataset: CMU-MOSEI mmdataset
    :param with_custom_split: whether we perform custom_split proposed by the paper written by Williams et al. (ACL 2018)
    :param pickle_name_fold: root name of the pickle object that contains the training, validation and test folds
    :param pickle_folder: name of the folder where the pickle object is saved
    :param image_feature: image feature type (either FACET 4.2 or OpenFace 2)
    :return: 3 lists train_list = [x_train, y_train, seg_train], valid_list = [x_valid, y_valid, seg_valid],
     test_list = [x_test, y_test, seg_test]
    """
    
    train_ids, valid_ids, test_ids = get_fold_ids(with_custom_split)
    if not pickle_file_exists(pickle_name_fold, pickle_folder, dataset_else_fold=False):
        train_list, valid_list, test_list = split_dataset(dataset,
                                                          train_ids, valid_ids, test_ids,
                                                          image_feature, pickle_name_fold, pickle_folder)
    else:
        train_list, valid_list, test_list = load_folds_from_pickle(pickle_name_fold, pickle_folder)
        
    return train_list, valid_list, test_list


# BUILD DATA GENERATOR AND TENSORFLOW DATASETS

def seq_with_fixed_length(seq_array, fixed_num_steps):
    """
    Pad or truncate sequence data to a given length.

    :param seq_array: 1 array of shape (own number of steps, number features)
    :param fixed_num_steps: fixed number of steps
    :return: seq_array with the fixed number of steps
    """
    own_num_steps = seq_array.shape[0]
    num_features = seq_array.shape[1]

    if fixed_num_steps >= own_num_steps:
        diff = fixed_num_steps - own_num_steps
        zeroes = np.zeros((diff, num_features))
        new_seq_array = np.concatenate((seq_array, zeroes))

    else:
        new_seq_array = seq_array[-fixed_num_steps:]

    return new_seq_array


def datapoint_generator(x_list, y_list, seg_list, with_fixed_length, fixed_num_steps):
    """
    Yields iteratively one datapoint represented by 1 array of features, 1 array of labels, 1 list of start/end times

    :param x_list: list of arrays of shape (number steps, number features)
    :param y_list: list of arrays of shape (1, 7), for the 7 emotions
    :param seg_list: list of ids of the segment described by (x, y). Example: 'zk2jTlAtvSU[1]'
    :param with_fixed_length: whether we fix all feature vectors to the same length
    :param fixed_num_steps: fixed number of steps in the sequence
    :return: yields 1 datapoint (x, y, seg_id) iteratively
    """

    if not len(x_list) == len(y_list) == len(seg_list):
        print("x_list, y_list, seg_list do not have the same number of elements")

    else:
        for x_list_i, y_list_i, seg_list_i in zip(x_list, y_list, seg_list):
            x_list_i = x_list_i if not with_fixed_length else seq_with_fixed_length(x_list_i, fixed_num_steps)
            x_list_i = x_list_i[None, :]
            yield x_list_i, y_list_i  #, seg_list_i


def get_tf_dataset(x_list, y_list, seg_list, batch_size, with_fixed_length, fixed_num_steps, train_mode):
    """
    Returns a TensorFlow dataset from a datapoint generator.
    #TODO Case where the number of steps is not fixed
    
    :param x_list: list of arrays of shape (number steps, number features)
    :param y_list: list of arrays of shape (1, 7), for the 7 emotions
    :param seg_list: list of ids of the segment described by (x, y). Example: 'zk2jTlAtvSU[1]'
    :param batch_size: batch size
    :param with_fixed_length: whether we fix all feature vectors to the same length
    :param fixed_num_steps: fixed number of steps in the sequence
    :return: a TensorFlow dataset
    """

    num_features = x_list[0].shape[1]
    tf_dataset = tf.data.Dataset.from_generator(generator=lambda: datapoint_generator(x_list, y_list, seg_list,
                                                                                      with_fixed_length,
                                                                                      fixed_num_steps),
                                                output_signature=(
                                                    tf.TensorSpec(shape=(None, fixed_num_steps, num_features),
                                                                  dtype=tf.float64),
                                                    tf.TensorSpec(shape=(1, 7), dtype=tf.float64)
                                                    # tf.TensorSpec(shape=(), dtype=tf.string)
                                                ))
    tf_dataset.shuffle(len(seg_list)).batch(batch_size).repeat() if train_mode else tf_dataset.batch(1).repeat(1)

    return tf_dataset
