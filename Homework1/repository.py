import gzip, cPickle, wget
import datetime
import os

MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
LOCAL_DATASET_PATH = "../data/mnist.pkl.gz"

def get_dataset():
    """Downloads the archive with the dataset if not present at the specified path.
       Opens the archive and returns training set, validation set and testing set as a tuple
    """

    if not os.path.isfile(LOCAL_DATASET_PATH) or not os.access(LOCAL_DATASET_PATH, os.R_OK):
        wget.download(MNIST_URL, LOCAL_DATASET_PATH)

    f = gzip.open(LOCAL_DATASET_PATH, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
         
    return train_set, valid_set, test_set

def save(model_name, classifier):
    now = datetime.datetime.now()
    with open("models/" + model_name + str(now.day) + str(now.hour) + str(now.minute), 'w+') as output:
        cPickle.dump(classifier, output, cPickle.HIGHEST_PROTOCOL)

def load(file_name):
    with open("models/" + file_name, 'rb') as input:
        print(cPickle.load(input))