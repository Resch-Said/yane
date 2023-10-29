from enum import Enum
import numpy as np

from src.neural_network.YaneConfig import get_binary_threshold


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return max(0, x)


def tanh(x):
    return np.tanh(x)


def linear(x):
    return x


def binary(x):
    return 1 if x > get_binary_threshold() else 0


class ActivationFunction(str, Enum):
    SIGMOID = "Sigmoid"
    RELU = "ReLU"
    TANH = "Tanh"
    LINEAR = "Linear"
    BINARY = "Binary"

    @classmethod
    def do_activation_function(cls, neuron):
        if neuron.activation_function == cls.SIGMOID:
            neuron.value = sigmoid(neuron.value)
        elif neuron.activation_function == cls.RELU:
            neuron.value = relu(neuron.value)
        elif neuron.activation_function == cls.TANH:
            neuron.value = tanh(neuron.value)
        elif neuron.activation_function == cls.LINEAR:
            neuron.value = linear(neuron.value)
        elif neuron.activation_function == cls.BINARY:
            neuron.value = binary(neuron.value)
        else:
            raise Exception("Unknown activation function")
