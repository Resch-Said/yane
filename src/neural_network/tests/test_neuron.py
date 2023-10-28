from src.neural_network.Neuron import Neuron
from src.neural_network.OutputNeuron import OutputNeuron


def test_init():
    neuron = Neuron(0)
    assert neuron.value == 0


def test_set_value():
    neuron = Neuron(0)
    neuron.value = 1
    assert neuron.value == 1


def test_set_value_negative():
    neuron = Neuron(0)
    neuron.value = -1
    assert neuron.value == -1


def test_set_value_float():
    neuron = Neuron(0)
    neuron.value = 0.5
    assert neuron.value == 0.5


def test_expected_value():
    neuron = OutputNeuron()
    neuron.expected_value = 1
    assert neuron.expected_value == 1


def test_expected_value_negative():
    neuron = OutputNeuron()
    neuron.expected_value = -1
    assert neuron.expected_value == -1


def test_expected_value_float():
    neuron = OutputNeuron()
    neuron.expected_value = 0.5
    assert neuron.expected_value == 0.5
