import pickle


def train_images():
    return pickle.load(open(r"D:\Users\Coding Projects\Documents\Project Mimir\Code Testing\Medium Extractor\cifar-10-batches-py\mnist_training_data.pickle", "rb"))


def test_images():
    return pickle.load(open(r"D:\Users\Coding Projects\Documents\Project Mimir\Code Testing\Medium Extractor\cifar-10-batches-py\mnist_testing_data.pickle", "rb"))


def train_labels():
    return pickle.load(open(r"D:\Users\Coding Projects\Documents\Project Mimir\Code Testing\Medium Extractor\cifar-10-batches-py\mnist_training_labels.pickle", "rb"))


def test_labels():
    return pickle.load(open(r"D:\Users\Coding Projects\Documents\Project Mimir\Code Testing\Medium Extractor\cifar-10-batches-py\mnist_testing_labels.pickle", "rb"))

