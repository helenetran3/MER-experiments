import os
import pickle


def pickle_file_exists(pickle_name, pickle_folder):
    """
    Check whether a pickle file exists.

    :param pickle_name: name of the pickle object
    :param pickle_folder: name of the folder where the pickle object is saved
    :return: True if the pickle file exists, else False
    """

    pickle_name_to_check = pickle_name + ".pkl"
    pickle_path = os.path.join(pickle_folder, pickle_name_to_check)

    return os.path.exists(pickle_path)


def save_with_pickle(obj_to_save, pickle_name, pickle_folder):
    """
    Save an object in a pickle file.

    :param obj_to_save: Object to be saved
    :param pickle_name: Name of the pickle object
    :param pickle_folder: Name of the folder where the pickle object is saved
    """

    if not os.path.isdir(pickle_folder):
        os.mkdir(pickle_folder)

    pickle_name += ".pkl"
    pickle_path = os.path.join(pickle_folder, pickle_name)

    with open(pickle_path, 'wb') as fw:
        pickle.dump(obj_to_save, fw)


def load_from_pickle(pickle_name, pickle_folder):
    """
    Load object from pickle file.

    :param pickle_name: name of the pickle object
    :param pickle_folder: name of the folder where the pickle object is saved
    :return: the object saved in the pickle file
    """

    pickle_name += ".pkl"
    pickle_path = os.path.join(pickle_folder, pickle_name)

    with open(pickle_path, 'rb') as fr:
        p_object = pickle.load(fr)

    return p_object


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
