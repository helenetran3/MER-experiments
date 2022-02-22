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

def download_dataset(pickle_name, pickle_folder="cmu_mosei", align_text=True, align_label=True):
    """
    Download CMU-MOSEI dataset using the SDK and perform data alignment (if desired).

    :param pickle_name: name of the pickle object that contains the CMU-MOSEI mmdataset
    :param pickle_folder: name of the folder where to save the pickle object
    :param align_text: whether we want data to align to the textual modality
    :param align_label: whether we want data to align to the labels
    """

    pickle_name_data = pickle_name + ".pkl"
    pickle_path_data = os.path.join(pickle_folder, pickle_name_data)

    if os.path.exists(pickle_path_data):
        print("{} already exists in {} folder. Please change the pickle name.".format(pickle_name_data, pickle_folder))

    else:
        if os.path.exists('cmu_mosei'):
            # Retrieve data and labels from cmu_mosei folder
            cmu_mosei = mmdatasdk.mmdataset('cmu_mosei/data/')
            cmu_mosei_labels = mmdatasdk.mmdataset('cmu_mosei/labels/')
        else:
            # Download data and labels and add them to cmu_mosei folder
            cmu_mosei = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.highlevel, 'cmu_mosei/data/')
            cmu_mosei_labels = mmdatasdk.mmdataset(mmdatasdk.cmu_mosei.labels, 'cmu_mosei/labels/')

        if align_text:
            cmu_mosei.align('glove_vectors', collapse_functions=[avg_collapse_function])

        if align_label:
            cmu_mosei.add_computational_sequences(mmdatasdk.cmu_mosei.labels, 'cmu_mosei/')
            cmu_mosei.align('Opinion Segment Labels')

        # Save cmu_mosei mmdataset with pickle
        if len(pickle_name) > 0:
            with open(pickle_path_data, 'wb') as fd:
                pickle.dump(cmu_mosei, fd)

        # Save cmu_mosei label mmdataset with pickle
        pickle_name_labels = pickle_name + "_labels.pkl"
        pickle_path_labels = os.path.join(pickle_folder, pickle_name_labels)
        with open(pickle_path_labels, 'wb') as fl:
            pickle.dump(cmu_mosei_labels, fl)


def load_dataset_pickle(pickle_name, pickle_folder="cmu_mosei"):
    """
    Load CMU-MOSEI data and labels from pickle file.

    :param pickle_name: name of the pickle object that contains the CMU-MOSEI mmdataset
    :param pickle_folder: name of the folder where to save the pickle object
    :return: cmu_mosei: mmdataset object containing all data (aligned or not)
             cmu_mosei_labels: mmdataset object containing all data labels
    """

    pickle_name_data = pickle_name + ".pkl"
    pickle_name_labels = pickle_name + "_labels.pkl"
    pickle_path_data = os.path.join(pickle_folder, pickle_name_data)
    pickle_path_labels = os.path.join(pickle_folder, pickle_name_labels)

    with open(pickle_path_data, 'rb') as fd1:
        cmu_mosei = pickle.load(fd1)

    with open(pickle_path_labels, 'rb') as fl1:
        cmu_mosei_labels = pickle.load(fl1)

    return cmu_mosei, cmu_mosei_labels


download_dataset("mytest")
