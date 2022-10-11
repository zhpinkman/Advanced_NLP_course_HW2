import numpy as np
from IPython import embed
from abc import ABC, abstractmethod
from typing import List


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


def tanh(x: np.array) -> np.array:
    return np.tanh(x)


def d_tanh(x: np.array, incoming_grad: np.array) -> np.array:
    dx = 1. - np.square(np.tanh(x))
    return np.multiply(dx, incoming_grad)


def relu(x: np.array) -> np.array:
    return np.multiply(x, x > 0)


def clip(gradient: np.array, max_value: float) -> np.array:
    np.clip(gradient, -max_value, max_value, out=gradient)
    return gradient


def d_relu(x: np.array, incoming_grad: np.array):
    # let's have g = Relu(x),
    # then dg is the incoming_grad and we want the dx to be returned
    # dx depending on the input (x) can be 1 or 0
    # if x > 0, the derivative
    mask = np.ones_like(x)
    dx = np.multiply(mask, x > 0)
    return np.multiply(incoming_grad, dx)


def sigmoid(x: np.array) -> np.array:
    return 1/(1 + np.exp(-x))


def d_sigmoid(x: np.array, incoming_grad: np.array) -> np.array:
    sigma = sigmoid(x)
    dx = np.multiply(sigma, 1 - sigma)
    return np.multiply(dx, incoming_grad)


def softmax(x: np.array) -> np.array:
    return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)


class DropoutLayer(NNComp):
    def __init__(self, prob: float = 0) -> None:
        self.prob = prob
        self.mask = None

    def forward(self, x):
        self.mask = (np.random.rand(*x.shape) > self.prob).astype(int)
        return np.multiply(self.mask, x) / (1.0 - self.prob)

    def backward(self, incoming_grad):
        return np.multiply(incoming_grad, self.mask)


class FeedForwardNetwork(NNComp):
    """
    This is one design option you might consider. Though it's
    solely a suggestion, and you are by no means required to stick
    to it. You can implement your FeedForwardNetwork as concrete
    implementations of the NNComp class, and fill forward and backward
    methods for each module accordingly. It will likely be composed of
    other NNComp objects.
    """

    def __init__(self, num_hiddens: List[int], weight_decay: float, max_seq_len: int, num_features: int, num_labels: int, dropout: float = 0):
        self.num_hiddens = num_hiddens
        self.weight_decay = weight_decay
        self.max_seq_len = max_seq_len
        self.num_features = num_features
        self.num_labels = num_labels
        self.activation_func = sigmoid
        self.d_activation_func = d_sigmoid
        self.dropout = dropout
        self.training = True

        # Weight matrices

        self.params = dict()
        np.random.seed(77)
        self.params['W1'] = np.random.randn(
            num_features * max_seq_len, self.num_hiddens[0]) * np.sqrt(2/(num_features * max_seq_len))
        self.params['b1'] = np.zeros(shape=[1, self.num_hiddens[0]])
        self.params[f'dropout1'] = DropoutLayer(self.dropout)

        for i in range(1, len(self.num_hiddens)):
            index = i + 1
            np.random.seed(66 + index)
            self.params[f"W{index}"] = np.random.randn(
                self.num_hiddens[i - 1], self.num_hiddens[i]) * np.sqrt(2/(self.num_hiddens[i - 1]))
            self.params[f"b{index}"] = np.zeros(shape=[1, self.num_hiddens[i]])
            self.params[f'dropout{index}'] = DropoutLayer(self.dropout)

        np.random.seed(55)
        self.params[f"W{len(self.num_hiddens) + 1}"] = np.random.randn(
            self.num_hiddens[-1], num_labels) * np.sqrt(2/self.num_hiddens[-1])
        self.params[f"b{len(self.num_hiddens) + 1}"] = np.zeros(
            shape=[1, num_labels])

        # To compute the gradient in the backward pass

        self.params['x'] = None
        for i in range(len(self.num_hiddens) + 1):
            index = i + 1
            self.params[f"z{index}"] = None
            self.params[f"a{index}"] = None

    def train(self):
        self.training = True

    def eval(self):
        self.training = False

    def forward(self, x: np.array):
        self.params['x'] = x

        self.params["z1"] = x @ self.params["W1"] + self.params["b1"]
        self.params["a1"] = self.activation_func(
            self.params["z1"]
        )
        # if self.training:
        #     self.params["a1"] = self.params['dropout1'].forward(
        #         self.params["a1"])

        for i in range(1, len(self.num_hiddens)):
            index = i + 1
            self.params[f"z{index}"] = self.params[f"a{index - 1}"] @ self.params[f"W{index}"] + \
                self.params[f"b{index}"]
            self.params[f"a{index}"] = self.activation_func(
                self.params[f"z{index}"])
            # if self.training:
            #     self.params[f"a{index}"] = self.params[f'dropout{index}'].forward(
            #         self.params[f"a{index}"]
            #     )

        self.params[f"z{len(self.num_hiddens) + 1}"] = self.params[f"a{len(self.num_hiddens)}"] @ self.params[f"W{len(self.num_hiddens) + 1}"] + \
            self.params[f"b{len(self.num_hiddens) + 1}"]
        self.params[f"a{len(self.num_hiddens) + 1}"] = softmax(
            self.params[f"z{len(self.num_hiddens) + 1}"])
        return self.params[f"a{len(self.num_hiddens) + 1}"]

    def backward(self, incoming_grad):
        # incoming_grad being dZ{len(self.num_hiddens) + 1}
        self.params[f"dz{len(self.num_hiddens) + 1}"] = incoming_grad
        num_examples = incoming_grad.shape[0]

        for i in range(len(self.num_hiddens), 0, -1):
            index = i + 1
            self.params[f"dW{index}"] = (
                self.params[f"a{index - 1}"].T @ self.params[f"dz{index}"]) / num_examples + self.params[f"W{index}"] / num_examples * self.weight_decay
            self.params[f"db{index}"] = np.mean(
                self.params[f"dz{index}"], axis=0, keepdims=True)
            self.params[f"da{index - 1}"] = self.params[f"dz{index}"] @ self.params[f"W{index}"].T
            # if self.training:
            # self.params[f"dz{index - 1}"] = self.d_activation_func(
            #     self.params[f"z{index - 1}"],
            #     self.params[f"dropout{index - 1}"].backward(
            #         self.params[f"da{index - 1}"]
            #     )
            # )
            # else:
            self.params[f"dz{index - 1}"] = self.d_activation_func(
                self.params[f"z{index - 1}"],
                self.params[f"da{index - 1}"]
            )

        self.params["dW1"] = (self.params["x"].T @
                              self.params["dz1"]) / num_examples + self.params["W1"] / num_examples * self.weight_decay
        self.params["db1"] = np.mean(self.params["dz1"], axis=0, keepdims=True)

    def update_weights(self, learning_rate, clip_value=5):
        for i in range(len(self.num_hiddens) + 1):
            index = i + 1
            self.params[f"W{index}"] -= learning_rate * \
                clip(self.params[f"dW{index}"], max_value=clip_value)
            self.params[f"b{index}"] -= learning_rate * \
                clip(self.params[f"db{index}"], max_value=clip_value)
