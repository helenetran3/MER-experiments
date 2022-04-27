import os
import pickle


def pickle_file_exists(pickle_name, root_folder):
    """
    Check whether a pickle file exists.

    :param pickle_name: name of the pickle object
    :param root_folder: name of the folder where the pickle object is saved
    :return: True if the pickle file exists, else False
    """

    pickle_name_to_check = pickle_name + ".pkl"
    pickle_path = os.path.join(root_folder, "pickle_files", pickle_name_to_check)

    return os.path.exists(pickle_path)


def save_with_pickle(obj_to_save, pickle_name, root_folder):
    """
    Save an object in a pickle file.

    :param obj_to_save: Object to be saved
    :param pickle_name: Name of the pickle object
    :param root_folder: Name of the folder where the pickle object is saved
    """

    pickle_folder = os.path.join(root_folder, "pickle_files")
    if not os.path.isdir(pickle_folder):
        os.makedirs(pickle_folder)

    pickle_name += ".pkl"
    pickle_path = os.path.join(root_folder, "pickle_files", pickle_name)

    with open(pickle_path, 'wb') as fw:
        pickle.dump(obj_to_save, fw)


def load_from_pickle(pickle_name, root_folder):
    """
    Load object from pickle file.

    :param pickle_name: name of the pickle object
    :param root_folder: name of the folder where the pickle object is saved
    :return: the object saved in the pickle file
    """

    pickle_name += ".pkl"
    pickle_path = os.path.join(root_folder, "pickle_files", pickle_name)

    with open(pickle_path, 'rb') as fr:
        p_object = pickle.load(fr)

    return p_object


def create_extension_name(predict_sentiment, predict_neutral_class):
    """
    Create an extension name which gives values for predict_sentiment and predict_neutral_class.

    :param predict_sentiment: whether we predict the sentiment
    :param predict_neutral_class: whether we predict neutral class
    :return: ext_name
    """

    ext_name = "_emo" if not predict_sentiment else ""
    ext_name = ext_name + "_with_n" if predict_neutral_class else ext_name

    return ext_name
