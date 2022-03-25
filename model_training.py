import os.path

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Dropout, Dense, LSTM
from dataset_utils import split_dataset, get_dataset

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam


def build_model(num_features, num_steps, num_layers, num_nodes, dropout_rate, final_activ):
    """
    Build the model described in the paper (cf. README for the reference).
    Works only when the number of steps is the same for all datapoints.

    :param num_features: feature vector size (it should be the same at each step)
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
        model.add(Dense(6, activation=final_activ))

    if num_layers == 2:
        model.add(BatchNormalization(input_shape=(num_steps, num_features)))
        model.add(Bidirectional(LSTM(64, return_sequences=True, input_shape=(num_steps, num_features))))
        model.add(Dropout(dropout_rate))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dropout(dropout_rate))
        model.add(Dense(num_nodes, activation="relu"))
        model.add(Dropout(dropout_rate))
        model.add(Dense(6, activation=final_activ))

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
        model.add(Dense(6, activation=final_activ))

    return model


def train_model(train_list, valid_list,
                batch_size, num_epochs, fixed_num_steps, num_layers,
                num_nodes, dropout_rate, final_activ, learning_rate, loss_function,
                val_metric, patience, model_dir, model_name):
    """
    Train the model.

    :param train_list: [x_train, y_train, seg_train] where x_train is a list of arrays of shape (number steps, number
    features), y_train a list arrays of shape (1, 7), and seg_train a list of segment ids (ex: 'zk2jTlAtvSU[1]')
    :param valid_list: [x_valid, y_valid, seg_valid]
    :param batch_size: Batch size for training
    :param num_epochs: Maximum number of epochs for training
    :param fixed_num_steps: Fixed size for all the sequences (if we keep the original size, this parameter is set to 0)
    :param num_layers: Number of bidirectional layers for the model
    :param num_nodes: Number of nodes for the penultimate dense layer
    :param dropout_rate: Dropout rate before each dense layer
    :param final_activ: Final activation function
    :param learning_rate: Learning rate for training
    :param loss_function: Loss function
    :param val_metric: Metric on validation data to monitor
    :param patience: Number of epochs with no improvement after which the training will be stopped
    :param model_dir: Name of the directory where the models will be saved
    :param model_name: Name of the model to be saved
    :return: history of the model training
    """

    x_train = train_list[0]
    y_train = train_list[1]
    seg_train = train_list[2]
    x_valid = valid_list[0]
    y_valid = valid_list[1]
    seg_valid = valid_list[2]
    num_train_samples = len(y_train)
    num_valid_samples = len(y_valid)

    # Create TensorFlow datasets for model training
    with_fixed_length = (fixed_num_steps > 0)
    train_dataset = get_dataset(x_train, y_train, seg_train, batch_size, with_fixed_length, fixed_num_steps)
    valid_dataset = get_dataset(x_valid, y_valid, seg_valid, batch_size, with_fixed_length, fixed_num_steps)

    # Parameters to save model
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    model_save_name = "{}_l_{}_n_{}_d_{}_b_{}_s_{}.h5".format(model_name, num_layers, num_nodes, dropout_rate,
                                                              batch_size, fixed_num_steps)
    model_save_path = os.path.join(model_dir, model_save_name)

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
    model = build_model(num_features, fixed_num_steps, num_layers, num_nodes, dropout_rate, final_activ)
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(loss=loss_function, optimizer=optimizer)

    history = model.fit(x=train_dataset,
                        epochs=num_epochs,
                        verbose=1,
                        steps_per_epoch=num_train_samples // batch_size,
                        validation_data=valid_dataset,
                        validation_steps=num_valid_samples // batch_size,
                        callbacks=[checkpoint, early_stopping])

    return history
