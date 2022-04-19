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


def save_results_in_csv_file(model_name, num_layers, num_nodes, dropout_rate, batch_size, fixed_num_steps,
                             loss_function, loss_function_val, metrics_regression, metrics_presence, metrics_dominant,
                             predict_neutral_class):
    """
    Save results of a single model to a csv file.

    :param model_name: Name of the model currently tested
    :param num_layers: Number of bidirectional layers for the model
    :param num_nodes: Number of nodes for the penultimate dense layer
    :param dropout_rate: Dropout rate before each dense layer
    :param batch_size: Batch size for training
    :param fixed_num_steps: Fixed size for all the sequences (if we keep the original size, this parameter is set to 0)
    :param loss_function: Loss function
    :param loss_function_val: Loss function obtained by the model
    :param metrics_regression: List of metrics for regression (w.r.t the presence scores)
    :param metrics_presence: List of metrics for detecting the presence/absence of an emotion
    :param metrics_dominant: List of metrics for detecting the dominant emotion(s)
    :param predict_neutral_class: Whether we predict the neutral class
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

    # Create csv folder
    model_csv_folder = os.path.join('models', model_name, 'csv')
    if not os.path.isdir(model_csv_folder):
        os.mkdir(model_csv_folder)

    # Create filenames
    csv_path_regression = os.path.join('models', model_name, 'csv', "regression.csv")
    csv_path_presence = os.path.join('models', model_name, 'csv', "classification_presence.csv")
    csv_path_dominant = os.path.join('models', model_name, 'csv', "classification_dominant.csv")

    # Create headers for metrics of each emotion
    metrics = ['f1', 'rec', 'prec', 'roc_auc']
    emotions = ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']
    if predict_neutral_class:
        emotions.append('neutral')
    header_presence_per_emotion = ['pres_{}_{}'.format(m, e) for m in metrics for e in emotions]
    header_dominant_per_emotion = ['dom_{}_{}'.format(m, e) for m in metrics for e in emotions]

    # Create headers for global metrics
    metrics_overall = ['acc', 'f1_unweighted', 'f1_weighted', 'rec_unweighted', 'rec_weighted', 'prec_unweighted', 
                       'prec_weighted', 'roc_auc_unweighted', 'roc_auc_weighted']
    header_presence_overall = ['pres_{}'.format(m) for m in metrics_overall]
    header_dominant_overall = ['dom_{}'.format(m) for m in metrics_overall]

    # Create the whole header for each csv file
    header_param = ['num_layers', 'num_nodes', 'dropout_rate', 'batch_size', 'fixed_num_steps', 'with_neutral_class',
                    loss_function]
    header_regression = header_param + ['mae', 'mse']
    header_presence = header_param + header_presence_overall + header_presence_per_emotion
    header_dominant = header_param + header_dominant_overall + header_dominant_per_emotion

    # Create data rows
    data_param = [num_layers, num_nodes, dropout_rate, batch_size, fixed_num_steps, predict_neutral_class,
                  loss_function_val]
    data_regression = data_param + metrics_regression
    data_presence = data_param + metrics_presence
    data_dominant = data_param + metrics_dominant

    # print(len(header_regression), len(data_regression))
    # print(len(header_dominant), len(data_dominant))
    # print(len(header_presence), len(data_presence))

    # Write in csv files
    write_csv(csv_path_regression, header_regression, data_regression)
    write_csv(csv_path_presence, header_presence, data_presence)
    write_csv(csv_path_dominant, header_dominant, data_dominant)

