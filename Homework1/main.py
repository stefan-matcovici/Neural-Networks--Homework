import cPickle, gzip, os
import numpy
import wget
import scipy.misc
import matplotlib.pyplot as plt

MNIST_URL = "http://deeplearning.net/data/mnist/mnist.pkl.gz"
LOCAL_DATASET_PATH = "../data/mnist.pkl.gz"

def get_dataset(path):
    """Downloads the archive with the dataset if not present at the specified path.
       Opens the archive and returns training set, validation set and testing set as a tuple
    """

    if not os.path.isfile(path) or not os.access(path, os.R_OK):
        wget.download(MNIST_URL, path)

    f = gzip.open(path, 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()
         
    return train_set, valid_set, test_set

if __name__ == "__main__":
    train_set, valid_set, test_set = get_dataset(LOCAL_DATASET_PATH)
    plt.imshow(train_set[0][0].reshape((28, 28)))
    plt.show()

    