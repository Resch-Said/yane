from enum import Enum
import numpy as np


class ActivationType(Enum):
    LINEAR = "linear"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    RELU = "relu"
    BINARY = "binary"


class ActivationFunction:

    @classmethod
    def activate(cls, activation_function, value):
        if activation_function == ActivationType.LINEAR:
            return cls.linear(value)
        elif activation_function == ActivationType.SIGMOID:
            return cls.sigmoid(value)
        elif activation_function == ActivationType.TANH:
            return cls.tanh(value)
        elif activation_function == ActivationType.RELU:
            return cls.relu(value)
        elif activation_function == ActivationType.BINARY:
            return cls.binary(value)
        else:
            raise Exception("Activation function not found")

    @staticmethod
    def linear(value):
        return value

    @staticmethod
    def sigmoid(value):
        return 1 / (1 + np.exp(-value))

    @staticmethod
    def tanh(value):
        return np.tanh(value)

    @staticmethod
    def relu(value):
        return np.maximum(0, value)

    @staticmethod
    def binary(value):
        if value >= 0.5:
            return 1
        else:
            return 0
