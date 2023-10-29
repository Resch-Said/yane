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


class ActivationFunction(Enum):
    SIGMOID = 0
    RELU = 1
    TANH = 2
    LINEAR = 3
    BINARY = 4
