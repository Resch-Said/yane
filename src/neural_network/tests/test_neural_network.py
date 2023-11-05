import pytest

from src.neural_network.Connection import Connection
from src.neural_network.HiddenNeuron import HiddenNeuron
from src.neural_network.InputNeuron import InputNeuron
from src.neural_network.NeuralNetwork import NeuralNetwork
from src.neural_network.Neuron import Neuron
from src.neural_network.OutputNeuron import OutputNeuron
from src.neural_network.exceptions.InvalidNeuron import InvalidNeuron


def test_get_all_neurons():
    nn = NeuralNetwork()

    nn.add_neuron(InputNeuron())
    nn.add_neuron(InputNeuron())
    nn.add_neuron(HiddenNeuron())
    nn.add_neuron(OutputNeuron())
    nn.add_neuron(OutputNeuron())
    nn.add_neuron(OutputNeuron())

    assert len(nn.get_all_neurons()) == 6
    assert len(nn.get_input_neurons()) == 2
    assert len(nn.get_hidden_neurons()) == 1
    assert len(nn.get_output_neurons()) == 3

    for neuron in nn.get_all_neurons():
        for neuron2 in nn.get_all_neurons():
            if neuron == neuron2:
                continue
            assert neuron.get_id() != neuron2.get_id()
            assert neuron.get_id() is not None
            assert neuron.get_id() != ""


def test_add_connection():
    nn = NeuralNetwork()
    assert len(nn.get_all_neurons()) == 0

    with pytest.raises(Exception):  # Raise because connection is empty
        nn.add_connection(Connection())

    con = Connection()
    con.set_in_neuron(InputNeuron())
    con.set_out_neuron(OutputNeuron())

    with pytest.raises(Exception):  # Raise because in_neuron is not in the network
        nn.add_connection(con)

    nn.add_neuron(con.get_in_neuron())
    with pytest.raises(Exception):  # Raise because out_neuron is not in the network
        nn.add_connection(con)

    nn.add_neuron(con.get_out_neuron())
    nn.add_connection(con)

    for neuron in nn.get_all_neurons():
        for con in neuron.get_next_connections():
            assert con.get_in_neuron() == neuron
            assert con.get_out_neuron() in nn.get_all_neurons()


def test_add_input_neuron():
    nn = NeuralNetwork()
    assert len(nn.get_all_neurons()) == 0

    nn.add_input_neuron(InputNeuron())
    assert len(nn.get_all_neurons()) == 1
    assert len(nn.get_input_neurons()) == 1


def test_add_hidden_neuron():
    nn = NeuralNetwork()
    assert len(nn.get_all_neurons()) == 0

    nn.add_hidden_neuron(HiddenNeuron())
    assert len(nn.get_all_neurons()) == 1
    assert len(nn.get_input_neurons()) == 0
    assert len(nn.get_hidden_neurons()) == 1


def test_add_output_neuron():
    nn = NeuralNetwork()
    assert len(nn.get_all_neurons()) == 0

    nn.add_output_neuron(OutputNeuron())
    assert len(nn.get_all_neurons()) == 1
    assert len(nn.get_input_neurons()) == 0
    assert len(nn.get_hidden_neurons()) == 0
    assert len(nn.get_output_neurons()) == 1


def test_add_neuron():
    nn = NeuralNetwork()
    assert len(nn.get_all_neurons()) == 0
    assert len(nn.get_input_neurons()) == 0
    assert len(nn.get_hidden_neurons()) == 0
    assert len(nn.get_output_neurons()) == 0

    new_neuron = InputNeuron()

    nn.add_neuron(InputNeuron())
    assert len(nn.get_all_neurons()) == 1
    assert len(nn.get_input_neurons()) == 1
    assert len(nn.get_hidden_neurons()) == 0
    assert len(nn.get_output_neurons()) == 0

    nn.add_neuron(InputNeuron())
    assert len(nn.get_all_neurons()) == 2

    nn.add_hidden_neuron(HiddenNeuron())
    assert len(nn.get_all_neurons()) == 3
    assert len(nn.get_input_neurons()) == 2
    assert len(nn.get_hidden_neurons()) == 1
    assert len(nn.get_output_neurons()) == 0

    nn.add_neuron(OutputNeuron())
    assert len(nn.get_all_neurons()) == 4
    assert len(nn.get_input_neurons()) == 2
    assert len(nn.get_hidden_neurons()) == 1
    assert len(nn.get_output_neurons()) == 1

    with pytest.raises(Exception):
        nn.add_neuron(Neuron())

    with pytest.raises(Exception):
        nn.add_input_neuron(Neuron())

    nn.add_input_neuron(InputNeuron())
    assert len(nn.get_all_neurons()) == 5
    assert len(nn.get_input_neurons()) == 3
    assert len(nn.get_hidden_neurons()) == 1
    assert len(nn.get_output_neurons()) == 1

    nn.add_hidden_neuron(HiddenNeuron())
    nn.add_output_neuron(OutputNeuron())

    with pytest.raises(Exception):
        nn.add_hidden_neuron(Neuron())

    with pytest.raises(Exception):
        nn.add_output_neuron(Neuron())

    assert len(nn.get_all_neurons()) == 7
    assert len(nn.get_input_neurons()) == 3
    assert len(nn.get_hidden_neurons()) == 2
    assert len(nn.get_output_neurons()) == 2

    for neuron in nn.get_all_neurons():
        for neuron2 in nn.get_all_neurons():
            if neuron == neuron2:
                continue
            assert neuron.get_id() != neuron2.get_id()
            assert neuron.get_id() is not None
            assert neuron.get_id() != ""

    nn.add_neuron(new_neuron)
    with pytest.raises(InvalidNeuron):
        nn.add_neuron(new_neuron)


def test_get_input_neurons():
    nn = NeuralNetwork()
    neuron = InputNeuron()
    neuron2 = InputNeuron()
    nn.add_input_neuron(neuron)
    nn.add_input_neuron(neuron2)

    assert len(nn.get_input_neurons()) == 2
    assert nn.get_input_neurons()[0].get_id() == neuron.get_id()
    assert nn.get_input_neurons()[1].get_id() == neuron2.get_id()


def test_get_hidden_neurons():
    nn = NeuralNetwork()
    neuron = HiddenNeuron()
    neuron2 = HiddenNeuron()
    nn.add_hidden_neuron(neuron)
    nn.add_hidden_neuron(neuron2)

    assert len(nn.get_hidden_neurons()) == 2
    assert nn.get_hidden_neurons()[0].get_id() == neuron.get_id()
    assert nn.get_hidden_neurons()[1].get_id() == neuron2.get_id()


def test_get_output_neurons():
    nn = NeuralNetwork()
    neuron = OutputNeuron()
    neuron2 = OutputNeuron()
    nn.add_output_neuron(neuron)
    nn.add_output_neuron(neuron2)

    assert len(nn.get_output_neurons()) == 2
    assert nn.get_output_neurons()[0].get_id() == neuron.get_id()
    assert nn.get_output_neurons()[1].get_id() == neuron2.get_id()


def test_get_neuron_by_id():
    nn = NeuralNetwork()
    neuron = InputNeuron()
    neuron2 = HiddenNeuron()
    neuron3 = OutputNeuron()
    nn.add_neuron(neuron)
    nn.add_neuron(neuron2)
    nn.add_neuron(neuron3)

    assert nn.get_neuron_by_id(neuron.get_id()) == neuron
    assert nn.get_neuron_by_id(neuron2.get_id()) == neuron2
    assert nn.get_neuron_by_id(neuron3.get_id()) == neuron3
    assert nn.get_neuron_by_id("invalid_id") is None

    assert nn.get_neuron_by_id(neuron.get_id()) != neuron2
    assert nn.get_neuron_by_id(neuron.get_id()) != neuron3
    assert nn.get_neuron_by_id(neuron2.get_id()) != neuron
    assert nn.get_neuron_by_id(neuron2.get_id()) != neuron3
    assert nn.get_neuron_by_id(neuron3.get_id()) != neuron
    assert nn.get_neuron_by_id(neuron3.get_id()) != neuron2


def test_get_all_connections():
    nn = NeuralNetwork()
    neuron = InputNeuron()
    neuron2 = HiddenNeuron()
    neuron3 = OutputNeuron()

    nn.add_neuron(neuron)
    nn.add_neuron(neuron2)
    nn.add_neuron(neuron3)

    nn.add_connection(Connection(neuron, neuron2))
    nn.add_connection(Connection(neuron2, neuron3))

    assert len(nn.get_all_connections()) == 2

    for con in nn.get_all_connections():
        assert con.get_in_neuron() in nn.get_all_neurons()
        assert con.get_out_neuron() in nn.get_all_neurons()
        assert con.get_in_neuron() != con.get_out_neuron()

        assert con.get_in_neuron().get_id() != con.get_out_neuron().get_id()
        assert con.get_in_neuron().get_id() is not None
        assert con.get_in_neuron().get_id() != ""
        assert con.get_out_neuron().get_id() is not None
        assert con.get_out_neuron().get_id() != ""

        assert con.get_in_neuron().get_id() == neuron.get_id() or con.get_in_neuron().get_id() == neuron2.get_id()
        assert con.get_out_neuron().get_id() == neuron2.get_id() or con.get_out_neuron().get_id() == neuron3.get_id()


def test_remove_neuron():
    nn = NeuralNetwork()
    neuron = InputNeuron()
    neuron2 = HiddenNeuron()
    neuron3 = OutputNeuron()
    nn.add_neuron(neuron)
    nn.add_neuron(neuron2)
    nn.add_neuron(neuron3)

    assert len(nn.get_all_neurons()) == 3
    assert len(nn.get_input_neurons()) == 1
    assert len(nn.get_hidden_neurons()) == 1
    assert len(nn.get_output_neurons()) == 1

    nn.remove_neuron(neuron)
    assert len(nn.get_all_neurons()) == 2
    assert len(nn.get_input_neurons()) == 0
    assert len(nn.get_hidden_neurons()) == 1
    assert len(nn.get_output_neurons()) == 1

    nn.remove_neuron(neuron2)
    assert len(nn.get_all_neurons()) == 1
    assert len(nn.get_input_neurons()) == 0
    assert len(nn.get_hidden_neurons()) == 0
    assert len(nn.get_output_neurons()) == 1

    nn.remove_neuron(neuron3)
    assert len(nn.get_all_neurons()) == 0
    assert len(nn.get_input_neurons()) == 0
    assert len(nn.get_hidden_neurons()) == 0
    assert len(nn.get_output_neurons()) == 0


def test_remove_neuron_with_connections():
    nn = NeuralNetwork()
    neuron1 = InputNeuron()
    neuron2 = HiddenNeuron()
    neuron3 = OutputNeuron()
    nn.add_neuron(neuron1)
    nn.add_neuron(neuron2)
    nn.add_neuron(neuron3)

    assert len(nn.get_all_neurons()) == 3
    assert len(nn.get_input_neurons()) == 1
    assert len(nn.get_hidden_neurons()) == 1
    assert len(nn.get_output_neurons()) == 1

    nn.add_connection(Connection(neuron1, neuron2))
    nn.add_connection(Connection(neuron2, neuron3))
    nn.add_connection(Connection(neuron1, neuron3))

    assert len(nn.get_all_connections()) == 3

    nn.remove_neuron(neuron2)
    assert len(nn.get_all_neurons()) == 2
    assert len(nn.get_input_neurons()) == 1
    assert len(nn.get_hidden_neurons()) == 0
    assert len(nn.get_output_neurons()) == 1

    assert len(nn.get_all_connections()) == 1  # 2 connections removed because neuron 2 had 2 connections

    # Check if all associations to neuron2 are removed
    for neuron in nn.get_all_neurons():
        for con in neuron.get_next_connections():
            assert con.get_in_neuron() != neuron2
            assert con.get_out_neuron() != neuron2

    nn.remove_neuron(neuron1)
    assert len(nn.get_all_neurons()) == 1
    assert len(nn.get_input_neurons()) == 0
    assert len(nn.get_hidden_neurons()) == 0
    assert len(nn.get_output_neurons()) == 1

    assert len(nn.get_all_connections()) == 0

    nn.remove_neuron(neuron3)
    assert len(nn.get_all_neurons()) == 0
    assert len(nn.get_input_neurons()) == 0
    assert len(nn.get_hidden_neurons()) == 0
    assert len(nn.get_output_neurons()) == 0

    assert len(nn.get_all_connections()) == 0


def test_set_input_data():
    assert False


def test_forward_propagation():
    assert False


def test_clear_values():
    assert False


def test_get_forward_order_list():
    assert False


def test_get_output_data():
    assert False


def test_evaluate():
    assert False


def test_custom_evaluation():
    assert False


def test_calculate_net_cost():
    assert False


def test_remove_all_connections():
    assert False


def test_mutate():
    assert False


def test_mutate_neuron_genes():
    assert False


def test_mutate_connection_genes():
    assert False


def test_add_random_connection():
    assert False


def test_get_random_neuron():
    assert False


def test_add_random_neuron():
    assert False
