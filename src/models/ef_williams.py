from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Bidirectional, Dropout, Dense, LSTM


class Model:
    """
    Williams, J., Kleinegesse, S., Comanescu, R., & Radu, O. (2018, July). Recognizing Emotions in Video Using
    Multimodal DNN Feature Fusion. In Proceedings of Grand Challenge and Workshop on Human Multimodal Language
    (Challenge-HML) (pp. 11-19).
    """

    def __init__(self, model_id, num_features, num_classes, num_steps, num_layers, num_nodes, dropout_rate, final_activ):
        """
        Build the model described in the paper (cf. README for the reference).
        Works only when the number of steps is the same for all datapoints.

        :param model_id: Model id (int)
        :param num_features: feature vector size (it should be the same at each step)
        :param num_classes: number of classes to predict
        :param num_steps: number of steps in the sequence
        :param num_layers: number of bidirectional layers
        :param num_nodes: number of nodes for the penultimate dense layer
        :param dropout_rate: dropout rate before each dense layer
        :param final_activ: final activation function
        """

        self.model_name = 'ef_williams'
        self.model_id = model_id
        self.num_features = num_features
        self.num_classes = num_classes
        self.num_steps = num_steps
        self.num_layers = num_layers
        self.num_nodes = num_nodes
        self.dropout_rate = dropout_rate
        self.final_activ = final_activ

    def build(self):
        """
        :return: the model built
        """

        model = Sequential()
        model.add(BatchNormalization(input_shape=(self.num_steps, self.num_features)))

        all_layers = range(self.num_layers)
        for i in all_layers:
            return_seq = True if i != all_layers[-1] else False  # Return sequence for all except the last layer
            model.add(Bidirectional(LSTM(64, return_sequences=return_seq), input_shape=(self.num_steps, self.num_features)))
            model.add(Dropout(self.dropout_rate))

        model.add(Dense(self.num_nodes, activation="relu"))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(self.num_classes, activation=self.final_activ))

        return model

    def print_values(self):
        print("Model name:", self.model_name)
        print("Model id:", self.model_id)
        print("Fixed number of steps:", self.num_steps)
        print("Number layers:", self.num_layers)
        print("Number nodes for the penultimate dense layer:", self.num_nodes)
        print("Dropout rate:", self.dropout_rate)
        print("Final activation:", self.final_activ)


def data_for_csv(model_name, num_layers, num_nodes, dropout_rate, final_activ):

    data = [model_name, num_layers, num_nodes, dropout_rate, final_activ]
    header = ['model_name', 'num_layers', 'num_nodes', 'dropout_rate', 'final_activ']

    return header, data
