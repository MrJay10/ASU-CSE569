import numpy as np


def load_samples(file_path):
    samples = []
    with open(file_path, "rb") as f:
        for line in f.readlines():
            x1, x2 = map(float, line.decode('UTF-8').strip('\n').split())
            samples.append((x1, x2))

    return np.array(samples)


def normalize_samples(samples):
    return (samples - samples.mean(axis=0))/samples.std(axis=0)


def load_data(data_file, label, normalize=True):
    x = load_samples(data_file)
    if normalize:
        x = normalize_samples(x)

    y = None
    if label == 0:
        y = np.zeros((x.shape[0], 1), dtype='int')
    elif label == 1:
        y = np.ones((x.shape[0], 1), dtype='int')

    return np.hstack((x, y))


class DataLoader:
    __instance = None

    @staticmethod
    def get_instance():
        if DataLoader.__instance is None:
            DataLoader()
        return DataLoader.__instance

    def __init__(self):
        if DataLoader.__instance is not None:
            raise Exception("Singleton Class can't be instantiated.")
        else:
            DataLoader.__instance = self

        self.train = None
        self.val = None
        self.test = None

        self.load_train_and_val()
        self.load_test()

    def load_train_and_val(self, normalized=True):
        train_1 = load_data(data_file="../Data/Train1.txt", label=0, normalize=normalized)
        train_2 = load_data(data_file="../Data/Train2.txt", label=1, normalize=normalized)

        self.train = np.vstack((train_1[:1500, :], train_2[:1500, :]))
        self.val = np.vstack((train_1[1500:, :], train_2[1500:, :]))

    def load_test(self, normalized=True):
        test_1 = load_data(data_file="../Data/Test1.txt", label=0, normalize=normalized)
        test_2 = load_data(data_file="../Data/Test2.txt", label=1, normalize=normalized)

        self.test = np.vstack((test_1, test_2))

    def get_training_data(self):
        return self.train

    def get_validation_data(self):
        return self.val

    def get_testing_data(self):
        return self.test
