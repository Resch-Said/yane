import pytest

from src.neural_network import YaneConfig
from src.neural_network.ActivationFunction import ActivationFunction
from src.neural_network.Connection import Connection
from src.neural_network.InputNeuron import InputNeuron
from src.neural_network.Neuron import Neuron
from src.neural_network.exceptions.InvalidConnection import InvalidConnection


def test_set_value():
    neuron = Neuron()
    neuron.set_value(10)
    assert neuron.get_value() == 10


def test_set_activation():
    neuron = Neuron()
    neuron.set_activation(ActivationFunction.SIGMOID)
    assert neuron.get_activation() == ActivationFunction.SIGMOID


def test_set_activation_2():
    neuron = Neuron()

    activation = YaneConfig.get_random_activation_function(YaneConfig.load_json_config())

    neuron.set_activation(ActivationFunction.get_function(activation))
    assert neuron.get_activation() == ActivationFunction.get_function(activation)


def test_get_value():
    neuron = Neuron()
    neuron.set_value(10)
    assert neuron.get_value() == 10


def test_get_activation():
    neuron = Neuron()
    neuron.set_activation(ActivationFunction.SIGMOID)
    assert neuron.get_activation() == ActivationFunction.SIGMOID


def test_get_next_connections():
    neuron = Neuron()
    connection = Connection()
    connection.set_in_neuron(neuron)
    connection.set_out_neuron(neuron)
    neuron.add_next_connection(connection)

    assert connection in neuron.get_next_connections()

    connection2 = Connection()
    assert connection2 not in neuron.get_next_connections()
    connection = Connection()
    assert connection not in neuron.get_next_connections()


def test_add_next_connection():
    neuron = Neuron()
    connection = Connection()
    with pytest.raises(InvalidConnection):
        neuron.add_next_connection(connection)

    connection.set_in_neuron(neuron)
    with pytest.raises(InvalidConnection):
        neuron.add_next_connection(connection)

    connection.set_out_neuron(neuron)
    neuron.add_next_connection(connection)

    assert connection in neuron.get_next_connections()

    with pytest.raises(InvalidConnection):
        neuron.add_next_connection(connection)

    connection2 = Connection()
    connection2.set_in_neuron(neuron)
    connection2.set_out_neuron(neuron)
    with pytest.raises(InvalidConnection):
        neuron.add_next_connection(connection2)

    connection2.set_in_neuron(Neuron())
    with pytest.raises(InvalidConnection):
        neuron.add_next_connection(connection2)

    connection2.set_in_neuron(neuron)
    connection2.set_out_neuron(Neuron())
    neuron.add_next_connection(connection2)

    assert connection2 in neuron.get_next_connections()
    assert connection in neuron.get_next_connections()
    connection.set_out_neuron(connection2.get_out_neuron())
    with pytest.raises(InvalidConnection):
        neuron.add_next_connection(connection)


def test_activate():
    neuron = Neuron()
    value = 10
    neuron.set_value(value)
    neuron.activate()

    assert neuron.get_value() == ActivationFunction.activate(neuron.get_activation(), value)


def test_reset():
    neuron = Neuron()
    neuron.set_value(10)
    neuron.reset()
    assert neuron.get_value() == 0.0


def test_get_id():
    Neuron.ID = 0

    neuron = Neuron()
    assert neuron.get_id() == 0
    neuron = Neuron()
    assert neuron.get_id() == 1
    neuron = Neuron()
    assert neuron.get_id() == 2


def test_copy():
    neuron = Neuron()
    neuron.set_value(10)
    neuron.set_activation(ActivationFunction.SIGMOID)

    neuron_copy = neuron.copy()

    assert neuron.get_value() == neuron_copy.get_value()
    assert neuron.get_activation() == neuron_copy.get_activation()

    neuron_copy.set_value(20)
    neuron_copy.set_activation(ActivationFunction.RELU)

    assert neuron.get_value() != neuron_copy.get_value()
    assert neuron.get_activation() != neuron_copy.get_activation()


def test_fire():
    neuron_from = Neuron()
    neuron_to = Neuron()

    neuron_from.set_value(10)
    neuron_from.set_activation(ActivationFunction.LINEAR)

    neuron_to.set_value(3)
    neuron_to.set_activation(ActivationFunction.LINEAR)

    connection = Connection()
    connection.set_in_neuron(neuron_from)
    connection.set_out_neuron(neuron_to)
    connection.set_weight(0.5)

    neuron_from.add_next_connection(connection)
    neuron_from.fire()

    assert neuron_to.get_value() == 3.0 + 10.0 * 0.5

    neuron_from.set_activation(ActivationFunction.SIGMOID)
    neuron_to.set_value(3)

    neuron_from.fire()

    assert neuron_to.get_value() == 3.0 + ActivationFunction.activate(ActivationFunction.SIGMOID, 10.0) * 0.5
    assert neuron_to.get_value() != 3.0 + 10.0 * 0.5

    neuron_to.set_value(3)
    neuron_to_2 = Neuron()
    neuron_to_2.set_value(4)

    connection_2 = Connection()
    connection_2.set_in_neuron(neuron_from)
    connection_2.set_out_neuron(neuron_to_2)
    connection_2.set_weight(2)

    neuron_from.add_next_connection(connection_2)
    neuron_from.set_value(10)
    neuron_from.fire()

    assert neuron_to.get_value() == 3.0 + ActivationFunction.activate(ActivationFunction.SIGMOID, 10.0) * 0.5
    assert neuron_to.get_value() != 3.0 + 10.0 * 0.5
    assert neuron_to_2.get_value() == 4.0 + ActivationFunction.activate(ActivationFunction.SIGMOID, 10.0) * 2.0


def test_mutate_activation_function():
    neuron = Neuron()
    neuron.set_activation(ActivationFunction.SIGMOID)
    while neuron.get_activation() == ActivationFunction.SIGMOID:
        neuron.mutate_activation_function()
    assert neuron.get_activation() != ActivationFunction.SIGMOID


def test_input_neuron():
    neuron = InputNeuron()
    assert neuron.get_activation() == ActivationFunction.LINEAR

    with pytest.raises(Exception):
        neuron.mutate_activation_function()

    with pytest.raises(Exception):
        neuron.set_activation(ActivationFunction.SIGMOID)


def test_wrong_connection():
    neuron = Neuron()
    neuron2 = Neuron()
    connection = Connection(neuron2, neuron)
    with pytest.raises(InvalidConnection):
        neuron.add_next_connection(connection)


def test_remove_connection():
    neuron = Neuron()
    neuron2 = Neuron()
    connection = Connection(neuron, neuron2)
    neuron.add_next_connection(connection)

    assert connection in neuron.get_next_connections()
    neuron.remove_next_connection(connection)
    assert connection not in neuron.get_next_connections()
