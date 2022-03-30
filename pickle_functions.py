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
