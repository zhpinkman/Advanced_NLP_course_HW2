import numpy as np
from IPython import embed
from abc import ABC, abstractmethod
import random
random.seed(77)
np.random.seed(77)


class NNComp(ABC):
    """
    This is one design option you might consider. Though it's
    solely a suggestion, and you are by no means required to stick
    to it. You can implement your NN modules as concrete
    implementations of this class, and fill forward and backward
    methods for each module accordingly.
    """

    @abstractmethod
    def forward(self, x):
        raise NotImplemented

    @abstractmethod
    def backward(self, incoming_grad):
        raise NotImplemented


def relu(x: np.array) -> np.array:
    return np.multiply(x, x > 0)


def softmax(x: np.array) -> np.array:
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


def d_relu(x: np.array, incoming_grad: np.array):
    # let's have g = Relu(x),
    # then dg is the incoming_grad and we want the dx to be returned
    # dx depending on the input (x) can be 1 or 0
    # if x > 0, the derivative
    mask = np.ones_like(x)
    dx = np.multiply(mask, x > 0)
    return np.multiply(incoming_grad, dx)


class FeedForwardNetwork(NNComp):
    """
    This is one design option you might consider. Though it's
    solely a suggestion, and you are by no means required to stick
    to it. You can implement your FeedForwardNetwork as concrete
    implementations of the NNComp class, and fill forward and backward
    methods for each module accordingly. It will likely be composed of
    other NNComp objects.
    """

    def __init__(self, num_hidden: int, max_seq_len: int, num_features: int, num_labels: int):
        self.num_hidden = num_hidden
        self.max_seq_len = max_seq_len
        self.num_features = num_features
        self.num_labels = num_labels

        # Weight matrices

        self.W1 = np.random.randn(
            num_features * max_seq_len, num_hidden) * np.sqrt(2/(num_features * max_seq_len))
        self.b1 = np.zeros(shape=[1, num_hidden])

        self.W2 = np.random.randn(
            num_hidden, num_labels) * np.sqrt(2/num_hidden)
        self.b2 = np.zeros(shape=[1, num_labels])

        # To compute the gradient in the backward pass
        self.x = None
        self.z1 = None
        self.a1 = None
        self.z2 = None
        self.a2 = None

    def forward(self, x: np.array):
        self.x = x
        z1 = x @ self.W1 + self.b1
        self.z1 = z1
        a1 = relu(z1)
        self.a1 = a1

        z2 = a1 @ self.W2 + self.b2
        self.z2 = z2

        a2 = softmax(z2)
        self.a2 = a2
        return a2

    def backward(self, incoming_grad):
        # incoming_grad being dZ2

        dW2 = self.a1.T @ incoming_grad
        db2 = np.mean(incoming_grad, axis=0, keepdims=True)

        da1 = incoming_grad @ self.W2.T
        dz1 = d_relu(self.z1, da1)

        dW1 = self.x.T @ dz1
        db1 = np.mean(dz1, axis=0, keepdims=True)

        derivatives = {
            'dW2': dW2,
            'db2': db2,
            'dW1': dW1,
            'db1': db1
        }
        return derivatives

    def update_weights(self, derivates, learning_rate):
        self.W1 -= learning_rate * derivates['dW1']
        self.b1 -= learning_rate * derivates['db1']
        self.W2 -= learning_rate * derivates['dW2']
        self.b2 -= learning_rate * derivates['db2']
