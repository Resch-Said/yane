import pytest

from src.neural_network.util.ActivationFunction import ActivationFunction


def test_get_function():
    assert ActivationFunction.get_function('linear') == ActivationFunction.LINEAR
    assert ActivationFunction.get_function('sigmoid') == ActivationFunction.SIGMOID
    assert ActivationFunction.get_function('tanh') == ActivationFunction.TANH
    assert ActivationFunction.get_function('relu') == ActivationFunction.RELU
    assert ActivationFunction.get_function('binary') == ActivationFunction.BINARY
    with pytest.raises(Exception):
        ActivationFunction.get_function('nonexistent')


def test_activate():
    assert ActivationFunction.activate(ActivationFunction.LINEAR, 1) == 1
    assert ActivationFunction.activate(ActivationFunction.SIGMOID, 0) == 0.5
    assert ActivationFunction.activate(ActivationFunction.TANH, 0) == 0
    assert ActivationFunction.activate(ActivationFunction.RELU, -1) == 0
    assert ActivationFunction.activate(ActivationFunction.BINARY, 0) in [0, 1]
    with pytest.raises(Exception):
        ActivationFunction.activate('nonexistent', 0)


def test_linear():
    assert ActivationFunction.linear(1) == 1


def test_sigmoid():
    assert ActivationFunction.sigmoid(0) == 0.5


def test_tanh():
    assert ActivationFunction.tanh(0) == 0


def test_relu():
    assert ActivationFunction.relu(-1) == 0


def test_binary():
    assert ActivationFunction.binary(0) in [0, 1]
