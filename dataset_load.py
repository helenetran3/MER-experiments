import os
import pickle
import numpy as np
from mmsdk import mmdatasdk


# COLLAPSE FUNCTIONS FOR DATA ALIGNMENT

def avg_collapse_function(intervals, features):
    """
    Collapse function using average for visual and vocal modality alignment to textual modality.
    """
    return np.average(features, axis=0)


# CMU-MOSEI DATA DOWNLOAD FUNCTIONS

def download_dataset(pickle_name, pickle_folder="cmu_mosei/pickle_files/", align_text=True, align_label=True):
    """
    Download CMU-MOSEI dataset using the SDK and perform data alignment (if desired).

    :param pickle_name: name of the pickle object that contains the CMU-MOSEI mmdataset
    :param pickle_folder: name of the folder where to save the pickle object
    :param align_text: whether we want data to align to the textual modality
    :param align_label: whether we want data to align to the labels
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

        if align_text:
            cmu_mosei.align('glove_vectors', collapse_functions=[avg_collapse_function])

        if align_label:
            cmu_mosei.add_computational_sequences(mmdatasdk.cmu_mosei.labels, 'cmu_mosei/')
            cmu_mosei.align('All Labels')

        # Save cmu_mosei mmdataset with pickle
        with open(pickle_path, 'wb') as fw:
            pickle.dump(cmu_mosei, fw)


def load_dataset_pickle(pickle_name, pickle_folder="cmu_mosei/pickle_files/"):
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
