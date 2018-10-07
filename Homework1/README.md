# Homework 1: Handwritten digits classifier
## Data
Training was done using the MNIST dataset available at : http://deeplearning.net/data/mnist/mnist.pkl.gz. This archive contains 3 sets:

* a training set of 50 000 elements. 
* a validation set of 10 000 elements
* a test set of 10 000 elements

Each set is composed of 2 arrays:

* The sample array is an array of subunitar numbers (0 represents white, 1 is black and anything in between is a shade of gray) of length 784 (corresponding to a 28x28 pixels image). 
* The label array is an array of digits corresponding to each sample in the first set

The [repository module](src/repository.py) downloads the dataset in the folder ../data if this file doesn't exist already.

## Training
