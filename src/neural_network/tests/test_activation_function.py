import pytest

from src.neural_network import YaneConfig
from src.neural_network.ActivationFunction import ActivationFunction
from src.neural_network.exceptions.InvalidActivation import InvalidActivation

yane_config = YaneConfig.load_json_config()


def test_linear():
    assert ActivationFunction.linear(0) == 0
    assert ActivationFunction.linear(1) == 1
    assert ActivationFunction.linear(-1) == -1
    assert ActivationFunction.linear(0.5) == 0.5


def test_sigmoid():
    assert ActivationFunction.sigmoid(0) == 0.5
    assert ActivationFunction.sigmoid(1) == 0.7310585786300049
    assert ActivationFunction.sigmoid(-1) == 0.2689414213699951


def test_tanh():
    assert ActivationFunction.tanh(0) == 0
    assert ActivationFunction.tanh(1) == 0.7615941559557649
    assert ActivationFunction.tanh(-1) == -0.7615941559557649


def test_relu():
    assert ActivationFunction.relu(0) == 0
    assert ActivationFunction.relu(1) == 1
    assert ActivationFunction.relu(-1) == 0


def test_binary():
    assert ActivationFunction.binary(0) == 0
    assert ActivationFunction.binary(1) == 1
    assert ActivationFunction.binary(-1) == 0
    assert ActivationFunction.binary(YaneConfig.get_binary_threshold(yane_config)) == 1


def test_get_function():
    assert ActivationFunction.get_function("linear") == ActivationFunction.LINEAR
    assert ActivationFunction.get_function("sigmoid") == ActivationFunction.SIGMOID
    assert ActivationFunction.get_function("tanh") == ActivationFunction.TANH
    assert ActivationFunction.get_function("relu") == ActivationFunction.RELU
    assert ActivationFunction.get_function("binary") == ActivationFunction.BINARY
    with pytest.raises(InvalidActivation):
        ActivationFunction.get_function("not_found")


def test_activate():
    assert ActivationFunction.activate(ActivationFunction.LINEAR, 0) == 0
    assert ActivationFunction.activate(ActivationFunction.SIGMOID, 0) == 0.5
    assert ActivationFunction.activate(ActivationFunction.TANH, 0) == 0
    assert ActivationFunction.activate(ActivationFunction.RELU, 0) == 0
    assert ActivationFunction.activate(ActivationFunction.BINARY, 0) == 0
    with pytest.raises(Exception):
        ActivationFunction.activate("not_found", 0)
