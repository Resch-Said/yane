from enum import Enum

import numpy as np

from src.neural_network import YaneConfig

yane_config = YaneConfig.load_json_config()


class ActivationFunction(str, Enum):
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    BINARY = "binary"

    @classmethod
    def get_function(cls, name):
        name = name.lower()
        if name == "linear":
            return cls.LINEAR
        elif name == "sigmoid":
            return cls.SIGMOID
        elif name == "tanh":
            return cls.TANH
        elif name == "relu":
            return cls.RELU
        elif name == "binary":
            return cls.BINARY
        else:
            raise Exception("Activation function not found")

    @classmethod
    def activate(cls, activation_function, value):
        if activation_function == cls.LINEAR:
            return cls.linear(value)
        elif activation_function == cls.SIGMOID:
            return cls.sigmoid(value)
        elif activation_function == cls.TANH:
            return cls.tanh(value)
        elif activation_function == cls.RELU:
            return cls.relu(value)
        elif activation_function == cls.BINARY:
            return cls.binary(value)
        else:
            raise Exception("Activation function not found")

    @classmethod
    def linear(cls, value):
        return value

    @classmethod
    def sigmoid(cls, value):
        return 1 / (1 + np.exp(-value))

    @classmethod
    def tanh(cls, value):
        return np.tanh(value)

    @classmethod
    def relu(cls, value):
        return np.maximum(0, value)

    @classmethod
    def binary(cls, value):
        if value > YaneConfig.get_binary_threshold(yane_config):
            return 1
        else:
            return 0
