import os.path

from src.pickle_functions import save_with_pickle
from src.dataset_utils import get_tf_dataset

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Dropout, Dense, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop


def get_optimizer(optimizer):
    """
    Get TensorFlow optimizer from string.

    :param optimizer: string
    :return: TensorFlow optimizer
    """

    opt_dict = {
        'adam': Adam,
        'sgd': SGD,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'rmsprop': RMSprop
    }

    if optimizer.lower() in opt_dict.keys():
        return opt_dict[optimizer]
    else:
        raise ValueError("Optimizer name '{}' not valid. Please choose among Adam, SGD, Adagrad, Adadelta and RMSprop."
                         .format(optimizer))


def create_model_folder_and_path(model_name, model_id):
    """
    Create model folder and path to file containing model weights.

    :param model_name: Name of the model
    :param model_id: Model id (int)
    :return: model_folder, model_save_path
    """

    model_folder = os.path.join('models_tested', model_name)
    if not os.path.isdir('models_tested'):
        os.mkdir('models_tested')
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    model_save_name = "model_{}.h5".format(model_id)
    model_save_path = os.path.join(model_folder, model_save_name)

    return model_folder, model_save_path


def build_model(num_features, num_classes, num_steps, num_layers, num_nodes, dropout_rate, final_activ):
    """
    Build the model described in the paper (cf. README for the reference).
    Works only when the number of steps is the same for all datapoints.

    :param num_features: feature vector size (it should be the same at each step)
    :param num_classes: number of classes to predict
    :param num_steps: number of steps in the sequence
    :param num_layers: number of bidirectional layers
    :param num_nodes: number of nodes for the penultimate dense layer
    :param dropout_rate: dropout rate before each dense layer
    :param final_activ: final activation function
    :return: the model built
    """

    model = Sequential()

    if num_layers == 1:
        model.add(BatchNormalization(input_shape=(num_steps, num_features)))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_nodes, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes, activation=final_activ))

    if num_layers == 2:
        model.add(BatchNormalization(input_shape=(num_steps, num_features)))
        model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(num_steps, num_features))))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_nodes, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes, activation=final_activ))

    if num_layers == 3:
        model.add(BatchNormalization(input_shape=(num_steps, num_features)))
        model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(num_steps, num_features))))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(num_steps, num_features))))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_nodes, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_classes, activation=final_activ))

    return model


def train_model(train_list, valid_list, test_list,
                batch_size, num_epochs, fixed_num_steps, num_layers,
                num_nodes, dropout_rate, final_activ, learning_rate, optimizer_tf, optimizer_name, loss_function,
                val_metric, patience, model_name, predict_neutral_class, model_id):
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
    :param optimizer_tf: TensorFlow optimizer
    :param optimizer_name: optimizer name
    :param loss_function: Loss function
    :param val_metric: Metric on validation data to monitor
    :param patience: Number of epochs with no improvement after which the training will be stopped
    :param model_name: Name of the model currently tested
    :param predict_neutral_class: Whether we predict the neutral class
    :param model_id: Model id (int)
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
    monitor = 'val_loss' if val_metric == 'loss' else 'val_accuracy'
    mode_monitor = 'min' if val_metric == 'loss' else 'max'

    # Initialize callbacks
    checkpoint = ModelCheckpoint(filepath=model_save_path,
                                 verbose=1,
                                 monitor=monitor,
                                 mode=mode_monitor,
                                 save_best_only=True)

    early_stopping = EarlyStopping(monitor=monitor,
                                   patience=patience,
                                   mode=mode_monitor,
                                   restore_best_weights=True)

    # Build model
    num_features = x_train[0].shape[1]
    model = build_model(num_features, num_classes, fixed_num_steps, num_layers, num_nodes, dropout_rate, final_activ)
    optimizer_lr = optimizer_tf(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer_lr)

    print("\n\n")
    print(model.summary())

    print("\n\n============================== Training Parameters ===========================================")
    print("\n>>> Dataset")
    print("Number training datapoints: {} ({:.2f}%)".format(num_train_samples, 100 * (num_train_samples / total_data)))
    print("Number validation datapoints: {} ({:.2f}%)".format(num_valid_samples, 100 * (num_valid_samples / total_data)))
    print("Number test datapoints: {} ({:.2f}%)".format(num_test_samples, 100 * (num_test_samples / total_data)))
    print("Number of classes:", num_classes)
    print("Predict neutral class:", predict_neutral_class)
    print("\n>>> Model parameters")
    print("Model id:", model_id)
    print("Model name:", model_name)
    print("Fixed number of steps:", fixed_num_steps)
    print("Number layers:", num_layers)
    print("Number nodes for the penultimate dense layer:", num_nodes)
    print("Dropout rate:", dropout_rate)
    print("Final activation:", final_activ)
    print("\n>>> Model training")
    print("Batch size:", batch_size)
    print("Number epochs:", num_epochs)
    print("Patience:", patience)
    print("Learning rate:", learning_rate)
    print("Optimizer:", optimizer_name)
    print("Loss function:", loss_function)
    print("Metric to monitor on validation data:", val_metric)

    print("\n\n================================= Model Training =============================================\n")
    history = model.fit(x=train_dataset,
                        epochs=num_epochs,
                        verbose=1,
                        steps_per_epoch=num_train_samples // batch_size,
                        validation_data=valid_dataset,
                        validation_steps=num_valid_samples // batch_size,
                        callbacks=[checkpoint, early_stopping])

    save_with_pickle(history, "history_{}".format(model_id), root_folder=model_folder)
