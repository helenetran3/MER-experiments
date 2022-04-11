import os
import csv
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


def save_results_in_csv_file(csv_name, csv_folder, num_layers, num_nodes, dropout_rate, batch_size, fixed_num_steps,
                             loss_function, loss_function_val, mae, mse,
                             acc, acc_bal, f1_micro, f1_macro, f1_weighted, rec_micro, rec_macro, rec_weighted):
                             # roc_auc_macro, roc_auc_weighted):
    """
    Save results of a single model to a csv file.

    :param csv_name: Name of the directory where the csv file containing the results is saved
    :param csv_folder: Name of the csv file
    :param num_layers: Number of bidirectional layers for the model
    :param num_nodes: Number of nodes for the penultimate dense layer
    :param dropout_rate: Dropout rate before each dense layer
    :param batch_size: Batch size for training
    :param fixed_num_steps: Fixed size for all the sequences (if we keep the original size, this parameter is set to 0)
    :param loss_function: Loss function
    :param loss_function_val: Loss function obtained by the model
    :param mae: Mean absolute error
    :param mse: Mean squared error
    :param acc: Accuracy
    :param acc_bal: Balanced accuracy
    :param f1_micro: F1 score (all classes together)
    :param f1_macro: F1 score (F1 score for each + unweighted mean)
    :param f1_weighted: F1 score (F1 score for each + weighted mean)
    :param rec_micro: Recall (all classes together)
    :param rec_macro: Recall (recall for each + unweighted mean)
    :param rec_weighted: Recall (recall for each + weighted mean)
    # :param roc_auc_macro: ROC AUC score (score for each + unweighted mean) - computes AUC of each class vs the rest
    # :param roc_auc_weighted: ROC AUC score (score for each + weighted mean) - computes AUC of each class vs the rest
    :return: One-line results added to the csv file.
    """

    csv_name += ".csv"
    csv_path = os.path.join(csv_folder, csv_name)

    data_to_save = [num_layers, num_nodes, dropout_rate, batch_size, fixed_num_steps, loss_function_val, mae, mse,
                    acc, acc_bal, f1_micro, f1_macro, f1_weighted, rec_micro, rec_macro, rec_weighted]
                    # roc_auc_macro, roc_auc_weighted]

    if not os.path.isdir(csv_folder):
        os.mkdir(csv_folder)

    if os.path.exists(csv_path):
        with open(csv_path, 'a+', newline='') as f:  # Open file in append mode
            writer = csv.writer(f)
            writer.writerow(data_to_save)

    else:
        with open(csv_path, 'w', encoding='UTF8') as f:  # Create file
            writer = csv.writer(f)
            header = ['num_layers', 'num_nodes', 'dropout_rate', 'batch_size', 'fixed_num_steps', loss_function, 'mae',
                      'mse', 'acc', 'acc_bal', 'f1_micro', 'f1_macro', 'f1_weighted', 'rec_micro', 'rec_macro',
                      'rec_weighted']  #, 'roc_auc_macro', 'roc_auc_weighted']
            writer.writerow(header)
            writer.writerow(data_to_save)

