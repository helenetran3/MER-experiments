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
                             loss_function, loss_function_val, mae, mse, metrics_presence, metrics_dominant,
                             predict_neutral_class):
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
    :return: One-line results added to the csv file.
    """

    def write_csv(csv_path, header, data_to_save):
        if os.path.exists(csv_path):
            with open(csv_path, 'a+', newline='') as f:  # Open file in append mode
                writer = csv.writer(f)
                writer.writerow(data_to_save)

        else:
            with open(csv_path, 'w', encoding='UTF8') as f:  # Create file
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerow(data_to_save)


    csv_name_regression = csv_name + "_regression.csv"
    csv_name_presence = csv_name + "_classif_presence.csv"
    csv_name_dominant = csv_name + "_classif_dominant.csv"
    csv_path_regression = os.path.join(csv_folder, csv_name_regression)
    csv_path_presence = os.path.join(csv_folder, csv_name_presence)
    csv_path_dominant = os.path.join(csv_folder, csv_name_dominant)

    # data_to_save = [num_layers, num_nodes, dropout_rate, batch_size, fixed_num_steps, loss_function_val, mae, mse]
    #                + metrics_presence + metrics_dominant

    if not os.path.isdir(csv_folder):
        os.mkdir(csv_folder)

    header_param = ['num_layers', 'num_nodes', 'dropout_rate', 'batch_size', 'fixed_num_steps', loss_function,
                    'with_neutral_class']
    header_regression = header_param + ['mae', 'mse']

    if predict_neutral_class:
        header_presence_metrics_per_emotion = ['pres_f1_happy', 'pres_f1_sad', 'pres_f1_anger',
                                               'pres_f1_surprise', 'pres_f1_disgust', 'pres_f1_fear',
                                               'pres_f1_neutral',
                                               'pres_rec_happy', 'pres_rec_sad', 'pres_rec_anger',
                                               'pres_rec_surprise', 'pres_rec_disgust', 'pres_rec_fear',
                                               'pres_rec_neutral',
                                               'pres_roc_auc_happy', 'pres_roc_auc_sad', 'pres_roc_auc_anger',
                                               'pres_roc_auc_surprise', 'pres_roc_auc_disgust', 'pres_roc_auc_fear',
                                               'pres_roc_auc_neutral']
        header_dominant_metrics_per_emotion = ['dom_f1_happy', 'dom_f1_sad', 'dom_f1_anger',
                                               'dom_f1_surprise', 'dom_f1_disgust', 'dom_f1_fear',
                                               'dom_f1_neutral',
                                               'dom_rec_happy', 'dom_rec_sad', 'dom_rec_anger',
                                               'dom_rec_surprise', 'dom_rec_disgust', 'dom_rec_fear',
                                               'dom_rec_neutral',
                                               'dom_roc_auc_happy', 'dom_roc_auc_sad', 'dom_roc_auc_anger',
                                               'dom_roc_auc_surprise', 'dom_roc_auc_disgust', 'dom_roc_auc_fear',
                                               'dom_roc_auc_neutral']
    else:
        header_presence_metrics_per_emotion = ['pres_f1_happy', 'pres_f1_sad', 'pres_f1_anger',
                                               'pres_f1_surprise', 'pres_f1_disgust', 'pres_f1_fear',
                                               'pres_f1_neutral',
                                               'pres_rec_happy', 'pres_rec_sad', 'pres_rec_anger',
                                               'pres_rec_surprise', 'pres_rec_disgust', 'pres_rec_fear',
                                               'pres_rec_neutral',
                                               'pres_roc_auc_happy', 'pres_roc_auc_sad', 'pres_roc_auc_anger',
                                               'pres_roc_auc_surprise', 'pres_roc_auc_disgust', 'pres_roc_auc_fear',
                                               'pres_roc_auc_neutral']
        header_dominant_metrics_per_emotion = ['dom_f1_happy', 'dom_f1_sad', 'dom_f1_anger',
                                               'dom_f1_surprise', 'dom_f1_disgust', 'dom_f1_fear',
                                               'dom_f1_neutral',
                                               'dom_rec_happy', 'dom_rec_sad', 'dom_rec_anger',
                                               'dom_rec_surprise', 'dom_rec_disgust', 'dom_rec_fear',
                                               'dom_rec_neutral',
                                               'dom_roc_auc_happy', 'dom_roc_auc_sad', 'dom_roc_auc_anger',
                                               'dom_roc_auc_surprise', 'dom_roc_auc_disgust', 'dom_roc_auc_fear',
                                               'dom_roc_auc_neutral']

    header_presence = header_param + ['pres_acc',
                                      'pres_f1_macro', 'pres_f1_weighted',
                                      'pres_rec_macro', 'pres_rec_weighted',
                                      'pres_roc_auc_macro', 'pres_roc_auc_weighted'] \
                      + header_presence_metrics_per_emotion
    header_dominant = header_param + ['dom_acc',
                                      'dom_f1_macro', 'dom_f1_weighted',
                                      'dom_rec_macro', 'dom_rec_weighted',
                                      'dom_roc_auc_macro', 'dom_roc_auc_weighted'] \
                      + header_dominant_metrics_per_emotion

    data_param = [num_layers, num_nodes, dropout_rate, batch_size, fixed_num_steps, loss_function_val,
                  predict_neutral_class]
    data_regression = data_param + [mae, mse]
    data_presence = data_param + metrics_presence
    data_dominant = data_param + metrics_dominant

    # print(len(header_regression), len(data_regression))
    # print(len(header_dominant), len(data_dominant))
    # print(len(header_presence), len(data_presence))

    write_csv(csv_path_regression, header_regression, data_regression)
    write_csv(csv_path_presence, header_presence, data_presence)
    write_csv(csv_path_dominant, header_dominant, data_dominant)

