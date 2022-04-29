import os.path

from src.pickle_functions import save_with_pickle
from src.dataset_utils import get_tf_dataset
from src.models import ef_williams

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from tensorflow.keras.utils import plot_model

from matplotlib import pyplot as plt


def get_optimizer(optimizer_str):
    """
    Get TensorFlow optimizer from string.

    :param optimizer_str: string
    :return: TensorFlow optimizer
    """

    opt_dict = {
        'adam': Adam,
        'sgd': SGD,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'rmsprop': RMSprop
    }

    if optimizer_str.lower() in opt_dict.keys():
        return opt_dict[optimizer_str]
    else:
        raise ValueError("Optimizer name '{}' not valid. Please choose among Adam, SGD, Adagrad, Adadelta and RMSprop."
                         .format(optimizer_str))


def get_model(model_str, model_id, num_features, num_classes, fixed_num_steps, num_layers, num_nodes, dropout_rate,
              final_activ):
    """
    Get model class from string.

    :param model_str: string
    :param model_id: Model id (int)
    :param num_features: Number of features
    :param num_classes: Number of classes
    :param fixed_num_steps: Fixed size for all the sequences (if we keep the original size, this parameter is set to 0)
    :param num_layers: Number of bidirectional layers for the model
    :param num_nodes: Number of nodes for the penultimate dense layer
    :param dropout_rate: Dropout rate before each dense layer
    :param final_activ: Final activation function
    :return: Model class
    """

    model_dict = {
        'ef_williams': ef_williams.Model(model_id, num_features, num_classes, fixed_num_steps, num_layers, num_nodes,
                                         dropout_rate, final_activ)
    }

    if model_str.lower() in model_dict.keys():
        return model_dict[model_str]

    else:
        raise ValueError("Model name '{}' not valid. Please choose among ef_williams.".format(model_str))


def create_model_folder_and_path(model_name, model_id):
    """
    Create model folder and path to file containing model weights.

    :param model_name: Name of the model
    :param model_id: Model id (int)
    :return: model_folder, model_save_path
    """

    model_folder = os.path.join('models_tested', model_name, 'models')
    if not os.path.isdir('models_tested'):
        os.makedirs(model_folder)
    model_save_name = "model_{}.h5".format(model_id)
    model_save_path = os.path.join(model_folder, model_save_name)

    return model_folder, model_save_path


def plot_history(history, model_id, val_metric, history_folder, root_folder, display_fig):
    """
    Plot history, display and save figures

    :param history: TensorFlow history object
    :param model_id: Model id (int)
    :param val_metric: Metric used to monitor validation data
    :param history_folder: Name of the folder where the history plot will be saved
    :param root_folder: Name of the root folder where the history folder is created
    :param display_fig: Whether we display the figures
    """

    # Create full path folder
    full_path_folder = os.path.join(root_folder, history_folder)
    if not os.path.isdir(full_path_folder):
        os.makedirs(full_path_folder)

    # summarize history for training and validation metric used to monitor
    plt.plot(history.history[val_metric])
    plt.plot(history.history['val_' + val_metric])
    plt.title('Model {}'.format(val_metric))
    plt.ylabel(val_metric)
    plt.xlabel('epoch')
    plt.legend(['train', 'val'])
    if display_fig:
        plt.show()
    history_plot_path = os.path.join(full_path_folder, 'history_{}_{}.png'.format(model_id, val_metric))
    plt.savefig(history_plot_path, bbox_inches='tight')


def train_model(train_list, valid_list, test_list,
                batch_size, num_epochs, fixed_num_steps, num_layers,
                num_nodes, dropout_rate, final_activ, learning_rate, optimizer_name, loss_function,
                val_metric, patience, model_name, predict_neutral_class, model_id, display_fig):
    """
    Train the model.

    :param train_list: [x_train, y_train, seg_train] where x_train is a list of arrays of shape (number steps, number
    features), y_train a list arrays of shape (1, 6 or 7), and seg_train a list of segment ids (ex: 'zk2jTlAtvSU[1]')
    :param valid_list: [x_valid, y_valid, seg_valid]
    :param test_list: [x_test, y_test, seg_test]
    :param batch_size: Batch size for training
    :param num_epochs: Maximum number of epochs for training
    :param fixed_num_steps: Fixed size for all the sequences (if we keep the original size, this parameter is set to 0)
    :param num_layers: Number of bidirectional layers for the model
    :param num_nodes: Number of nodes for the penultimate dense layer
    :param dropout_rate: Dropout rate before each dense layer
    :param final_activ: Final activation function
    :param learning_rate: Learning rate for training
    :param optimizer_name: optimizer name
    :param loss_function: Loss function
    :param val_metric: Metric on validation data to monitor
    :param patience: Number of epochs with no improvement after which the training will be stopped
    :param model_name: Name of the model currently tested
    :param predict_neutral_class: Whether we predict the neutral class
    :param model_id: Model id (int)
    :param display_fig: Whether we display the figures
    """

    x_train = train_list[0]
    y_train = train_list[1]
    seg_train = train_list[2]
    x_valid = valid_list[0]
    y_valid = valid_list[1]
    seg_valid = valid_list[2]
    num_classes = y_train[0].shape[1]
    num_train_samples = len(y_train)
    num_valid_samples = len(y_valid)
    num_test_samples = len(test_list[1])
    total_data = num_train_samples + num_valid_samples + num_test_samples

    # Create TensorFlow datasets for model training
    with_fixed_length = (fixed_num_steps > 0)
    train_dataset = get_tf_dataset(x_train, y_train, seg_train, num_classes, batch_size, with_fixed_length,
                                   fixed_num_steps, train_mode=True)
    valid_dataset = get_tf_dataset(x_valid, y_valid, seg_valid, num_classes, batch_size, with_fixed_length,
                                   fixed_num_steps, train_mode=True)

    # Parameters to save model
    model_folder, model_save_path = create_model_folder_and_path(model_name, model_id)

    # Parameters for metric monitoring
    val_monitor = 'val_loss' if val_metric == 'loss' else 'val_accuracy'
    mode_monitor = 'min' if val_metric == 'loss' else 'max'

    # Initialize callbacks
    checkpoint = ModelCheckpoint(filepath=model_save_path,
                                 verbose=1,
                                 monitor=val_monitor,
                                 mode=mode_monitor,
                                 save_best_only=True)

    early_stopping = EarlyStopping(monitor=val_monitor,
                                   patience=patience,
                                   mode=mode_monitor,
                                   restore_best_weights=True)

    # Build model
    num_features = x_train[0].shape[1]
    optimizer_tf = get_optimizer(optimizer_name)
    optimizer_lr = optimizer_tf(learning_rate=learning_rate)
    model_class = get_model(model_name, model_id, num_features, num_classes, fixed_num_steps, num_layers, num_nodes,
                            dropout_rate, final_activ)
    model = model_class.build()
    model.compile(loss=loss_function, optimizer=optimizer_lr)

    print("\n\n============================== Training Parameters ===========================================")
    print("\n>>> Dataset")
    print("Number training datapoints: {} ({:.2f}%)".format(num_train_samples, 100 * (num_train_samples / total_data)))
    print("Number validation datapoints: {} ({:.2f}%)".format(num_valid_samples, 100 * (num_valid_samples / total_data)))
    print("Number test datapoints: {} ({:.2f}%)".format(num_test_samples, 100 * (num_test_samples / total_data)))
    print("Number of output cells:", num_classes)
    print("Predict neutral class:", predict_neutral_class)
    print("\n>>> Model training")
    print("Batch size:", batch_size)
    print("Number epochs:", num_epochs)
    print("Patience:", patience)
    print("Learning rate:", learning_rate)
    print("Optimizer:", optimizer_name)
    print("Loss function:", loss_function)
    print("Metric to monitor on validation data:", val_metric)
    print("\n>>> Model parameters")
    model_class.print_values()
    print("\n")
    print(model.summary())

    print("\n\n================================= Model Training =============================================\n")
    history = model.fit(x=train_dataset,
                        epochs=num_epochs,
                        verbose=1,
                        steps_per_epoch=num_train_samples // batch_size,
                        validation_data=valid_dataset,
                        validation_steps=num_valid_samples // batch_size,
                        callbacks=[checkpoint, early_stopping])

    model_folder = os.path.join("models_tested", model_name)
    save_with_pickle(history, "history_{}".format(model_id), pickle_folder="history", root_folder=model_folder)

    # Save model plot
    plot_model(model, to_file=os.path.join(model_folder, "models", "model_{}.png".format(model_id)), show_shapes=True,
               show_layer_activations=True)

    # Display and save history plot
    plot_history(history, model_id, val_metric,
                 history_folder="history", root_folder=model_folder, display_fig=display_fig)
