import numpy as np
from DatasetBuilder import DataLoader
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import accuracy_score


class MultiLayerPerceptron:
    def __init__(self, n_input=2, n_hidden=2, n_output=1):

        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output

        self.training_data = None
        self.training_labels = None
        self.validation_data = None
        self.validation_labels = None
        self.testing_data = None
        self.testing_labels = None

        self.network = []
        self.__initialize_network()
        self.HIDDEN_LAYER_IDX = 0
        self.OUTPUT_LAYER_IDX = 1

    def __initialize_network(self):
        # +1 accounts for the weights from bias unit in input and hidden layers
        hidden_layer = [{'weights': np.random.rand(self.n_input+1)} for _ in range(self.n_hidden)]
        output_layer = [{'weights': np.random.rand(self.n_hidden+1)} for _ in range(self.n_output)]

        self.network.append(hidden_layer)
        self.network.append(output_layer)

    @staticmethod
    def __activate(value):
        return max(0, value)
        # return 1.0/(1.0 + np.exp(-value))

    def __forward_propagate(self, sample):
        # prefix 1 to simplify weighted sum with bias unit
        inputs = np.hstack(([1], sample[:]))

        for layer in self.network:
            outputs = []
            for neuron in layer:
                neuron['output'] = self.__activate(np.dot(neuron['weights'], inputs))
                outputs.append(neuron['output'])

            inputs = np.hstack(([1], outputs[:]))
        return inputs[1:]

    # def __transfer_derivative(self, value):
    #     return self.__activate(value) * (1.0 - self.__activate(value))

    @staticmethod
    def __transfer_derivative(value):
        return 1 if value > 0 else 0

    def __back_propagate(self, true_label):
        # begin with output layer
        output_layer = self.network[self.OUTPUT_LAYER_IDX]

        for neuron in output_layer:
            error = true_label - neuron['output']
            neuron['delta'] = error * self.__transfer_derivative(neuron['output'])

        # end with hidden layer
        hidden_layer = self.network[self.HIDDEN_LAYER_IDX]

        for n in range(len(hidden_layer)):
            error = 0.0
            for o_neuron in output_layer:
                error += o_neuron['weights'][n + 1] * o_neuron['delta']

            hidden_layer[n]['delta'] = error * self.__transfer_derivative(hidden_layer[n]['output'])

    def __update_weights(self, sample, lr):
        # prefix 1 to simplify weighted sum with bias unit
        inputs = np.hstack(([1], sample[:]))

        for layer in self.network:
            outputs = []
            for neuron in layer:
                neuron['weights'] += neuron['delta'] * inputs * lr
                outputs.append(neuron['output'])

            inputs = np.hstack(([1], outputs[:]))

    def __get_error(self, data, labels):
        total_error = 0.0
        for itr, sample in enumerate(data):
            output = self.__forward_propagate(sample)
            ground_truth = labels[itr]
            total_error += np.sum(np.square(output - ground_truth))

        return total_error/data.shape[0]

    def __get_validation_error(self):
        return self.__get_error(self.validation_data, self.validation_labels)

    def __get_testing_error(self):
        return self.__get_error(self.testing_data, self.testing_labels)

    def train(self, learning_rate=9e-2, n_epochs=1000, batch_size=-1):
        epoch = 0
        training_error_history = []
        validation_error = float('inf')
        validation_error_history = []
        testing_error_history = []

        stopping_condition_met = False
        while not stopping_condition_met:
            epoch += 1
            training_error = 0.0
            for itr, sample in enumerate(self.training_data):
                output = self.__forward_propagate(sample)
                ground_truth = self.training_labels[itr]
                training_error += np.sum(np.square(output - ground_truth))
                self.__back_propagate(ground_truth)
                self.__update_weights(sample, learning_rate)

            training_error = training_error/self.training_data.shape[0]
            updated_val_error = self.__get_validation_error()
            testing_error = self.__get_testing_error()

            stopping_condition_met = np.isclose(validation_error, updated_val_error) or epoch >= n_epochs
            validation_error = updated_val_error
            print("> epoch: {}; learning_rate: {}; training_error: {}; validation_error: {}"
                  .format(epoch, learning_rate, training_error, validation_error))
            training_error_history.append(training_error)
            validation_error_history.append(validation_error)
            testing_error_history.append(testing_error)
        print("Finished training")
        print()

        plt.title("MLP: {} x {} x {}".format(self.n_input, self.n_hidden, self.n_output))
        plt.xlabel("Epochs")
        plt.ylabel("J/n")
        plt.plot(training_error_history, label="Training")
        plt.plot(validation_error_history, label="Validation")
        plt.plot(testing_error_history, label="Testing")
        plt.legend()
        figure = plt.gcf()
        # plt.show()

        filename = './Plot_{}_{}_{}.png'.format(self.n_input, self.n_hidden, self.n_output)
        figure.savefig(filename, dpi=100)
        plt.close(figure)

    def predict(self, data):
        outputs = []
        for sample in data:
            outputs.append(np.round(self.__forward_propagate(sample)[0]))

        return np.array(outputs)

    def save_weights(self):
        filename = 'MLP_{}_{}_{}'.format(self.n_input, self.n_hidden, self.n_output)

        with open(filename, 'wb') as f:
            pickle.dump(self.network, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(filename + '.txt', 'w') as f:
            f.write(str(self.network))

    def load_weights(self):
        filename = 'MLP_{}_{}_{}'.format(self.n_input, self.n_output, self.n_hidden)

        with open(filename, 'rb') as f:
            self.network = pickle.load(f)

    def __str__(self):
        return "MLP: {} x {} x {}".format(self.n_input, self.n_hidden, self.n_output)


def train_and_test(mlp_classifier):
    # fetch and feed training data
    training_data = DataLoader.get_instance().get_training_data()
    np.random.shuffle(training_data)
    x_train = training_data[:, :2]
    y_train = training_data[:, 2]

    mlp_classifier.training_data = x_train
    mlp_classifier.training_labels = y_train

    # fetch and feed validation data
    validation_data = DataLoader.get_instance().get_validation_data()
    np.random.shuffle(validation_data)
    x_val = validation_data[:, :2]
    y_val = validation_data[:, 2]

    mlp_classifier.validation_data = x_val
    mlp_classifier.validation_labels = y_val

    # fetch and feed testing data
    testing_data = DataLoader.get_instance().get_testing_data()
    x_test = testing_data[:, :2]
    y_test = testing_data[:, 2]

    mlp_classifier.testing_data = x_test
    mlp_classifier.testing_labels = y_test

    # train
    mlp_classifier.train(learning_rate=9e-3)
    mlp_classifier.save_weights()

    y_pred = mlp_classifier.predict(x_test)
    res = accuracy_score(y_test, y_pred)
    print("Accuracy on testing data: {}".format(res))
    return res


if __name__ == '__main__':
    avg_scores = []
    for i in range(2, 10 + 1, 2):
        scores = []
        for _ in range(5):
            mlp = MultiLayerPerceptron(n_input=2, n_hidden=i, n_output=1)
            score = train_and_test(mlp)
            scores.append(score)
        avg_scores.append(np.mean(scores))

    plt.bar(range(1, len(avg_scores) + 1), avg_scores)
    plt.show()

    with open('scores.txt', 'w') as f:
        f.write(str(avg_scores))
