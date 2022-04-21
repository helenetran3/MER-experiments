import os
import csv
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

    if not os.path.isdir(root_folder):
        os.mkdir(root_folder)
    if not os.path.isdir(pickle_folder):
        os.mkdir(pickle_folder)

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


def save_results_in_csv_file(model_name, num_layers, num_nodes, dropout_rate, batch_size, fixed_num_steps,
                             loss_function, loss_function_val, metrics_regression, metrics_score_coa,
                             metrics_presence, metrics_dominant, predict_neutral_class, threshold_emo_pres):
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
    :param metrics_score_coa: List of metrics for classifying the presence score of each emotion
    :param metrics_presence: T lists of lists of metrics for detecting the presence/absence of an emotion (T is the number of thresholds set)
    :param metrics_dominant: List of metrics for detecting the dominant emotion(s)
    :param predict_neutral_class: Whether we predict the neutral class
    :param threshold_emo_pres: list of thresholds at which emotions are considered to be present. Must be between 0 and 3
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
    model_csv_folder = os.path.join('models_tested', model_name, 'csv')
    if not os.path.isdir(model_csv_folder):
        os.mkdir(model_csv_folder)

    # Create filenames
    csv_path_regression = os.path.join('models_tested', model_name, 'csv', "regression.csv")
    csv_path_score_coa = os.path.join('models_tested', model_name, 'csv', "classification_score_coarse.csv")
    csv_path_presence = [os.path.join('models_tested', model_name, 'csv', "classification_presence_t_{}.csv".format(thres))
                         for thres in threshold_emo_pres]
    csv_path_dominant = os.path.join('models_tested', model_name, 'csv', "classification_dominant.csv")

    # Create headers for metrics of each emotion
    metrics_emo_pres = ['f1', 'rec', 'prec']
    metrics_emo_all = metrics_emo_pres + ['roc_auc']
    emotions = ['happy', 'sad', 'anger', 'surprise', 'disgust', 'fear']
    if predict_neutral_class:
        emotions.append('neutral')
    header_score_coa_per_emotion = ['sc_coa_{}_{}'.format(m, e) for m in metrics_emo_all for e in emotions]
    header_presence_per_emotion = ['pres_{}_{}'.format(m, e) for m in metrics_emo_pres for e in emotions]
    header_dominant_per_emotion = ['dom_{}_{}'.format(m, e) for m in metrics_emo_all for e in emotions]

    # Create headers for global metrics
    metrics_overall_pres = ['acc', 'f1_unweighted', 'f1_weighted', 'rec_unweighted', 'rec_weighted', 'prec_unweighted',
                            'prec_weighted']
    metrics_overall_all = metrics_overall_pres + ['roc_auc_unweighted', 'roc_auc_weighted']
    header_score_coa_overall = ['sc_coa_{}'.format(m) for m in metrics_overall_all]
    header_presence_overall = ['pres_{}'.format(m) for m in metrics_overall_pres]
    header_dominant_overall = ['dom_{}'.format(m) for m in metrics_overall_all]

    # Create the whole header for each csv file
    header_param = ['num_layers', 'num_nodes', 'dropout_rate', 'batch_size', 'fixed_num_steps', 'with_neutral_class',
                    loss_function]
    header_regression = header_param + ['mae', 'mse']
    header_score_coa = header_param + header_score_coa_overall + header_score_coa_per_emotion
    header_presence = header_param + header_presence_overall + header_presence_per_emotion
    header_dominant = header_param + header_dominant_overall + header_dominant_per_emotion

    # Create data rows
    data_param = [num_layers, num_nodes, dropout_rate, batch_size, fixed_num_steps, predict_neutral_class,
                  loss_function_val]
    data_regression = data_param + metrics_regression
    data_score_coa = data_param + metrics_score_coa
    data_presence = [data_param + mp for mp in metrics_presence]
    data_dominant = data_param + metrics_dominant

    # print(len(header_regression), len(data_regression))
    # print(len(header_score_coa), len(data_score_coa))
    # print(len(header_dominant), len(data_dominant))
    # for i in range(len(metrics_presence)):
    #     print(len(header_presence), len(data_presence[i]))

    # Write in csv files
    write_csv(csv_path_regression, header_regression, data_regression)
    write_csv(csv_path_score_coa, header_score_coa, data_score_coa)
    for path, data in zip(csv_path_presence, data_presence):
        write_csv(path, header_presence, data)
    write_csv(csv_path_dominant, header_dominant, data_dominant)

