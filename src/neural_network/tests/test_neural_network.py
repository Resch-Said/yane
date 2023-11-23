import pytest

from src.neural_network.ActivationFunction import ActivationFunction
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

    assert len(nn.get_all_neurons()) == 7
    assert len(nn.get_input_neurons()) == 3
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
    assert len(nn.get_all_neurons()) == 1  # Bias neuron is added by default

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
    assert len(nn.get_all_neurons()) == 1

    nn.add_input_neuron(InputNeuron())
    assert len(nn.get_all_neurons()) == 2
    assert len(nn.get_input_neurons()) == 2


def test_add_hidden_neuron():
    nn = NeuralNetwork()
    assert len(nn.get_all_neurons()) == 1

    nn.add_hidden_neuron(HiddenNeuron())
    assert len(nn.get_all_neurons()) == 2
    assert len(nn.get_input_neurons()) == 1
    assert len(nn.get_hidden_neurons()) == 1


def test_add_output_neuron():
    nn = NeuralNetwork()
    assert len(nn.get_all_neurons()) == 1

    nn.add_output_neuron(OutputNeuron())
    assert len(nn.get_all_neurons()) == 2
    assert len(nn.get_input_neurons()) == 1
    assert len(nn.get_hidden_neurons()) == 0
    assert len(nn.get_output_neurons()) == 1


def test_add_neuron():
    nn = NeuralNetwork()
    assert len(nn.get_all_neurons()) == 1
    assert len(nn.get_input_neurons()) == 1
    assert len(nn.get_hidden_neurons()) == 0
    assert len(nn.get_output_neurons()) == 0

    new_neuron = InputNeuron()

    nn.add_neuron(InputNeuron())
    assert len(nn.get_all_neurons()) == 2
    assert len(nn.get_input_neurons()) == 2
    assert len(nn.get_hidden_neurons()) == 0
    assert len(nn.get_output_neurons()) == 0

    nn.add_neuron(InputNeuron())
    assert len(nn.get_all_neurons()) == 3

    nn.add_hidden_neuron(HiddenNeuron())
    assert len(nn.get_all_neurons()) == 4
    assert len(nn.get_input_neurons()) == 3
    assert len(nn.get_hidden_neurons()) == 1
    assert len(nn.get_output_neurons()) == 0

    nn.add_neuron(OutputNeuron())
    assert len(nn.get_all_neurons()) == 5
    assert len(nn.get_input_neurons()) == 3
    assert len(nn.get_hidden_neurons()) == 1
    assert len(nn.get_output_neurons()) == 1

    with pytest.raises(Exception):
        nn.add_neuron(Neuron())

    with pytest.raises(Exception):
        nn.add_input_neuron(Neuron())

    nn.add_input_neuron(InputNeuron())
    assert len(nn.get_all_neurons()) == 6
    assert len(nn.get_input_neurons()) == 4
    assert len(nn.get_hidden_neurons()) == 1
    assert len(nn.get_output_neurons()) == 1

    nn.add_hidden_neuron(HiddenNeuron())
    nn.add_output_neuron(OutputNeuron())

    with pytest.raises(Exception):
        nn.add_hidden_neuron(Neuron())

    with pytest.raises(Exception):
        nn.add_output_neuron(Neuron())

    assert len(nn.get_all_neurons()) == 8
    assert len(nn.get_input_neurons()) == 4
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

    assert len(nn.get_input_neurons()) == 3  # 2 input neurons + 1 bias neuron
    assert nn.get_input_neurons()[1].get_id() == neuron.get_id()
    assert nn.get_input_neurons()[2].get_id() == neuron2.get_id()


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

    assert len(nn.get_all_neurons()) == 4
    assert len(nn.get_input_neurons()) == 2
    assert len(nn.get_hidden_neurons()) == 1
    assert len(nn.get_output_neurons()) == 1

    nn.remove_neuron(neuron)
    assert len(nn.get_all_neurons()) == 3
    assert len(nn.get_input_neurons()) == 1
    assert len(nn.get_hidden_neurons()) == 1
    assert len(nn.get_output_neurons()) == 1

    nn.remove_neuron(neuron2)
    assert len(nn.get_all_neurons()) == 2
    assert len(nn.get_input_neurons()) == 1
    assert len(nn.get_hidden_neurons()) == 0
    assert len(nn.get_output_neurons()) == 1

    nn.remove_neuron(neuron3)
    assert len(nn.get_all_neurons()) == 1
    assert len(nn.get_input_neurons()) == 1
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

    assert len(nn.get_all_neurons()) == 4
    assert len(nn.get_input_neurons()) == 2
    assert len(nn.get_hidden_neurons()) == 1
    assert len(nn.get_output_neurons()) == 1

    nn.add_connection(Connection(neuron1, neuron2))
    nn.add_connection(Connection(neuron2, neuron3))
    nn.add_connection(Connection(neuron1, neuron3))

    assert len(nn.get_all_connections()) == 3

    nn.remove_neuron(neuron2)
    assert len(nn.get_all_neurons()) == 3
    assert len(nn.get_input_neurons()) == 2
    assert len(nn.get_hidden_neurons()) == 0
    assert len(nn.get_output_neurons()) == 1

    assert len(nn.get_all_connections()) == 1  # 2 connections removed because neuron 2 had 2 connections

    # Check if all associations to neuron2 are removed
    for neuron in nn.get_all_neurons():
        for con in neuron.get_next_connections():
            assert con.get_in_neuron() != neuron2
            assert con.get_out_neuron() != neuron2

    nn.remove_neuron(neuron1)
    assert len(nn.get_all_neurons()) == 2
    assert len(nn.get_input_neurons()) == 1
    assert len(nn.get_hidden_neurons()) == 0
    assert len(nn.get_output_neurons()) == 1

    assert len(nn.get_all_connections()) == 0

    nn.remove_neuron(neuron3)
    assert len(nn.get_all_neurons()) == 1
    assert len(nn.get_input_neurons()) == 1
    assert len(nn.get_hidden_neurons()) == 0
    assert len(nn.get_output_neurons()) == 0

    assert len(nn.get_all_connections()) == 0


def test_set_input_data():
    nn = NeuralNetwork()
    nn.set_input_data([2, 3, 4])
    assert nn.get_input_data() == [1, 2, 3, 4]
    assert len(nn.get_input_neurons()) == 4  # 3 input neurons + 1 bias neuron

    nn.set_input_data([0, 4, 6])
    assert nn.get_input_data() == [1, 0, 4, 6]
    assert len(nn.get_input_neurons()) == 4  # 3 input neurons + 1 bias neuron
    nn.set_input_data([1])
    assert nn.get_input_data() == [1, 1, 0, 0]
    nn.set_input_data([1, 2, 1, 2, 5])
    assert nn.get_input_data() == [1, 1, 2, 1, 2, 5]
    nn.set_input_data([])
    assert nn.get_input_data() == [1, 0, 0, 0, 0, 0]


def test_forward_propagation():
    nn = NeuralNetwork()
    input1 = InputNeuron()
    input2 = InputNeuron()
    hidden1 = HiddenNeuron()
    out1 = OutputNeuron()
    nn.add_neuron(input1)
    nn.add_neuron(input2)
    nn.add_neuron(hidden1)
    nn.add_neuron(out1)

    nn.add_connection(Connection(input1, hidden1))
    nn.add_connection(Connection(input2, hidden1))
    nn.add_connection(Connection(hidden1, out1))
    nn.set_input_data([1, 2])

    nn.forward_propagation()

    expected_output = input1.get_value() * input1.get_next_connections()[0].get_weight() + \
                      input2.get_value() * input2.get_next_connections()[0].get_weight()
    expected_output = ActivationFunction.activate(hidden1.get_activation(), expected_output)
    expected_output = expected_output * hidden1.get_next_connections()[0].get_weight()
    expected_output = ActivationFunction.activate(out1.get_activation(), expected_output)

    assert nn.get_output_data() == expected_output

    nn.forward_propagation()
    assert nn.get_output_data() == expected_output
    assert nn.get_output_data() == expected_output
    assert nn.get_input_data() == [1, 1, 2]


def test_forward_propagation_recurrent_connection():
    nn = NeuralNetwork()
    input1 = InputNeuron()
    input2 = InputNeuron()
    hidden1 = HiddenNeuron()
    out1 = OutputNeuron()
    nn.add_neuron(input1)
    nn.add_neuron(input2)
    nn.add_neuron(hidden1)
    nn.add_neuron(out1)

    nn.add_connection(Connection(input1, hidden1))
    nn.add_connection(Connection(input2, hidden1))
    nn.add_connection(Connection(hidden1, out1))
    nn.add_connection(Connection(out1, hidden1))

    nn.set_input_data([1, 2])

    nn.forward_propagation()

    expected_output = input1.get_value() * input1.get_next_connections()[0].get_weight() + \
                      input2.get_value() * input2.get_next_connections()[0].get_weight()
    expected_output = ActivationFunction.activate(hidden1.get_activation(), expected_output)
    expected_output = expected_output * hidden1.get_next_connections()[0].get_weight()
    expected_output = ActivationFunction.activate(out1.get_activation(), expected_output)

    assert nn.get_output_data() == expected_output

    recurrend_value = out1.get_value() * out1.get_next_connections()[0].get_weight()
    assert hidden1.get_value() == recurrend_value

    expected_output = input1.get_value() * input1.get_next_connections()[0].get_weight() + \
                      input2.get_value() * input2.get_next_connections()[0].get_weight() + recurrend_value
    expected_output = ActivationFunction.activate(hidden1.get_activation(), expected_output)
    expected_output = expected_output * hidden1.get_next_connections()[0].get_weight()
    expected_output = ActivationFunction.activate(out1.get_activation(), expected_output)

    nn.forward_propagation()

    assert nn.get_output_data() == expected_output
    assert nn.get_output_data() == expected_output
    assert nn.get_input_data() == [1, 1, 2]


def test_get_forward_order_list():
    nn = NeuralNetwork()
    input1 = InputNeuron()
    input2 = InputNeuron()
    hidden1 = HiddenNeuron()
    hidden2 = HiddenNeuron()
    out1 = OutputNeuron()
    nn.add_neuron(input1)
    nn.add_neuron(input2)
    nn.add_neuron(hidden1)
    nn.add_neuron(hidden2)
    nn.add_neuron(out1)

    nn.add_connection(Connection(input1, hidden1))
    nn.add_connection(Connection(input2, hidden1))
    nn.add_connection(Connection(hidden1, out1))
    nn.add_connection(Connection(out1, hidden1))

    forward_order = nn.get_forward_order_list()

    assert forward_order == [input1, input2, hidden1,
                             out1]  # Bias and hidden2 is not included because it has no connections


def test_calculate_net_cost():
    nn = NeuralNetwork()
    input1 = InputNeuron()
    input2 = InputNeuron()
    hidden1 = HiddenNeuron()
    hidden2 = HiddenNeuron()
    out1 = OutputNeuron()
    nn.add_neuron(input1)
    nn.add_neuron(input2)
    nn.add_neuron(hidden1)
    nn.add_neuron(hidden2)
    nn.add_neuron(out1)

    nn.add_connection(Connection(input1, hidden1))
    nn.add_connection(Connection(input2, hidden1))
    nn.add_connection(Connection(hidden1, out1))
    nn.add_connection(Connection(out1, hidden1))

    assert nn.calculate_net_cost() == 10


def test_remove_all_connections():
    nn = NeuralNetwork()
    input1 = InputNeuron()
    input2 = InputNeuron()
    hidden1 = HiddenNeuron()
    hidden2 = HiddenNeuron()
    out1 = OutputNeuron()
    nn.add_neuron(input1)
    nn.add_neuron(input2)
    nn.add_neuron(hidden1)
    nn.add_neuron(hidden2)
    nn.add_neuron(out1)

    nn.add_connection(Connection(input1, hidden1))
    nn.add_connection(Connection(input2, hidden1))
    nn.add_connection(Connection(hidden1, out1))
    nn.add_connection(Connection(out1, hidden1))

    assert len(nn.get_all_connections()) == 4

    nn.remove_all_connections()

    assert len(nn.get_all_connections()) == 0


def test_add_random_connection():
    nn = NeuralNetwork()
    input1 = InputNeuron()
    input2 = InputNeuron()
    hidden1 = HiddenNeuron()
    hidden2 = HiddenNeuron()
    out1 = OutputNeuron()
    nn.add_neuron(input1)
    nn.add_neuron(input2)
    nn.add_neuron(hidden1)
    nn.add_neuron(hidden2)
    nn.add_neuron(out1)

    nn.add_random_connection()
    assert len(nn.get_all_connections()) == 1
    nn.add_random_connection()
    nn.add_random_connection()
    nn.add_random_connection()
    nn.add_random_connection()
    nn.add_random_connection()
    nn.add_random_connection()

    assert len(nn.get_all_connections()) <= 7
    assert len(nn.get_all_connections()) >= 4


def test_get_random_neuron():
    nn = NeuralNetwork()
    input1 = InputNeuron()
    input2 = InputNeuron()
    hidden1 = HiddenNeuron()
    hidden2 = HiddenNeuron()
    out1 = OutputNeuron()
    nn.add_neuron(input1)
    nn.add_neuron(input2)
    nn.add_neuron(hidden1)
    nn.add_neuron(hidden2)
    nn.add_neuron(out1)

    assert nn.get_random_neuron() is not None
    assert nn.get_random_neuron() in nn.get_all_neurons()


def test_add_random_neuron():
    nn = NeuralNetwork()
    input1 = InputNeuron()
    out1 = OutputNeuron()
    nn.add_neuron(input1)
    nn.add_neuron(out1)

    nn.add_connection(Connection(input1, out1))

    assert len(nn.get_all_connections()) == 1

    new_neuron = nn.add_random_neuron()

    assert input1.next_connections[0].get_in_neuron() == input1
    assert input1.next_connections[0].get_out_neuron() == new_neuron
    assert new_neuron.next_connections[0].get_in_neuron() == new_neuron
    assert new_neuron.next_connections[0].get_out_neuron() == out1
    assert input1.next_connections[0].get_weight() == 1.0

    assert len(nn.get_all_connections()) == 2

    nn.print()

    assert len(nn.get_all_neurons()) == 4
    assert len(nn.get_input_neurons()) == 2
    assert len(nn.get_hidden_neurons()) == 1
    assert len(nn.get_output_neurons()) == 1
