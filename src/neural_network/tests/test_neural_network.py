from src.neural_network.HiddenNeuron import HiddenNeuron
from src.neural_network.InputNeuron import InputNeuron
from src.neural_network.NeuralNetwork import NeuralNetwork, mutate_weight
from src.neural_network.Neuron import Neuron
from src.neural_network.OutputNeuron import OutputNeuron


def test_neural_network_init():
    nn = NeuralNetwork()
    assert nn.hidden_neurons == []
    assert nn.connections == []


def test_add_input_neuron():
    nn = NeuralNetwork()
    neuron = Neuron()
    assert nn.input_neurons == []
    nn.add_input_neuron(neuron)
    assert nn.input_neurons == [neuron]


def test_add_hidden_neuron():
    nn = NeuralNetwork()
    neuron = Neuron()
    assert nn.hidden_neurons == []
    nn.add_hidden_neuron(neuron)
    assert nn.hidden_neurons == [neuron]


def test_add_output_neuron():
    nn = NeuralNetwork()
    neuron = Neuron()
    assert nn.output_neurons == []
    nn.add_output_neuron(neuron)
    assert nn.output_neurons == [neuron]


def test_remove_input_neuron():
    nn = NeuralNetwork()
    neuron = InputNeuron()
    assert nn.input_neurons == []
    nn.add_input_neuron(neuron)
    assert nn.input_neurons == [neuron]
    nn.remove_neuron(neuron)
    assert nn.input_neurons == []


def test_remove_hidden_neuron():
    nn = NeuralNetwork()
    neuron = HiddenNeuron()
    assert nn.hidden_neurons == []
    nn.add_hidden_neuron(neuron)
    assert nn.hidden_neurons == [neuron]
    nn.remove_neuron(neuron)
    assert nn.hidden_neurons == []


def test_remove_output_neuron():
    nn = NeuralNetwork()
    neuron = OutputNeuron()
    assert nn.output_neurons == []
    nn.add_output_neuron(neuron)
    assert nn.output_neurons == [neuron]
    nn.remove_neuron(neuron)
    assert nn.output_neurons == []


def test_add_connection_forward():
    nn = NeuralNetwork()
    neuron_from = Neuron()
    neuron_to = Neuron()
    assert nn.connections == []
    nn.add_connection(neuron_from, neuron_to)

    next_neurons = nn.get_connected_neurons_forward(neuron_from)
    assert next_neurons == [neuron_to]


def test_add_connection_backward():
    nn = NeuralNetwork()
    neuron_from = Neuron()
    neuron_to = Neuron()
    assert nn.connections == []
    nn.add_connection(neuron_from, neuron_to)

    next_neurons = nn.get_connected_neurons_backward(neuron_to)
    assert next_neurons == [neuron_from]


def test_remove_connection():
    nn = NeuralNetwork()
    neuron_from = Neuron()
    neuron_to = Neuron()
    assert nn.connections == []
    nn.add_connection(neuron_from, neuron_to)

    assert nn.connections != []
    nn.remove_connection(neuron_from, neuron_to)
    assert nn.connections == []


# TODO: Verhindern, dass Input und Output Neuronen gelöscht werden können
def test_remove_neuron_with_connections():
    nn = NeuralNetwork()
    neuron_from = InputNeuron()
    neuron_to = OutputNeuron()

    nn.add_input_neuron(neuron_from)
    nn.add_output_neuron(neuron_to)

    assert nn.connections == []
    assert len(nn.input_neurons) == 1
    assert len(nn.output_neurons) == 1
    nn.add_connection(neuron_from, neuron_to)

    assert nn.connections != []
    nn.remove_neuron(neuron_from)
    assert nn.connections == []
    assert nn.output_neurons == [neuron_to]
    assert len(nn.input_neurons) == 0
    assert len(nn.output_neurons) == 1


def test_forward_propagation():
    nn = NeuralNetwork()
    neuron_from1 = Neuron(value=2)
    neuron_from2 = Neuron(value=3)
    neuron_to = Neuron()
    nn.add_input_neuron(neuron_from1)
    nn.add_input_neuron(neuron_from2)
    nn.add_output_neuron(neuron_to)
    nn.add_connection(neuron_from1, neuron_to, 1)
    nn.add_connection(neuron_from2, neuron_to, 2)
    nn.forward_propagation()

    assert neuron_to.value == 8


def test_forward_propagation_2():
    nn = NeuralNetwork()
    neuron_input1 = Neuron(value=2)
    neuron_input2 = Neuron(value=4)
    neuron_output = Neuron()

    nn.add_input_neuron(neuron_input1)
    nn.add_input_neuron(neuron_input2)
    nn.add_output_neuron(neuron_output)

    nn.add_connection(neuron_input1, neuron_output, 1)
    nn.add_connection(neuron_input2, neuron_output, 0.5)
    nn.forward_propagation()

    # output_neuron = 2*1 + 4*0.5 = 4
    assert neuron_output.value == 4


def test_forward_propagation_with_multiple_layers():
    nn = NeuralNetwork()
    neuron_input1 = Neuron(value=2)
    neuron_input2 = Neuron(value=3)
    neuron_hidden = Neuron()
    neuron_output = Neuron()

    nn.add_input_neuron(neuron_input1)
    nn.add_input_neuron(neuron_input2)
    nn.add_hidden_neuron(neuron_hidden)
    nn.add_output_neuron(neuron_output)

    nn.add_connection(neuron_input1, neuron_hidden, 1)
    nn.add_connection(neuron_input2, neuron_hidden, 2)
    nn.add_connection(neuron_hidden, neuron_output, 4)

    # hidden_neuron = 2*1 + 3*2 = 8
    # output_neuron = 8*4 = 32
    nn.forward_propagation()

    assert neuron_hidden.value == 8
    assert neuron_output.value == 32


def test_forward_propagation_with_multiple_layers_and_multiple_outputs():
    nn = NeuralNetwork()
    neuron_input1 = Neuron(value=2)
    neuron_input2 = Neuron(value=3)
    neuron_hidden1 = Neuron()
    neuron_hidden2 = Neuron()
    neuron_output1 = Neuron()
    neuron_output2 = Neuron()

    nn.add_input_neuron(neuron_input1)
    nn.add_input_neuron(neuron_input2)
    nn.add_hidden_neuron(neuron_hidden1)
    nn.add_hidden_neuron(neuron_hidden2)
    nn.add_output_neuron(neuron_output1)
    nn.add_output_neuron(neuron_output2)

    nn.add_connection(neuron_input1, neuron_hidden1, 1)
    nn.add_connection(neuron_input2, neuron_hidden1, 2)
    nn.add_connection(neuron_input2, neuron_hidden2, 2)
    nn.add_connection(neuron_hidden1, neuron_output1, 4)
    nn.add_connection(neuron_hidden2, neuron_output1, 3)
    nn.add_connection(neuron_input2, neuron_output2, 5)
    nn.add_connection(neuron_hidden1, neuron_output2, 3)

    # hidden_neuron1 = 2*1 + 3*2 = 8
    # hidden_neuron2 = 3*2 = 6
    # output_neuron1 = 8*4 + 6*3 = 50
    # output_neuron2 = 3*5 + 8*3 = 39

    nn.forward_propagation()

    assert neuron_hidden1.value == 8
    assert neuron_hidden2.value == 6
    assert neuron_output1.value == 50
    assert neuron_output2.value == 39


def test_forward_propagation_with_multiple_layers_and_multiple_outputs_with_remove():
    nn = NeuralNetwork()
    neuron_input1 = Neuron(value=2)
    neuron_input2 = Neuron(value=3)
    neuron_hidden1 = Neuron()
    neuron_hidden2 = Neuron()
    neuron_output1 = Neuron()
    neuron_output2 = Neuron()

    nn.add_input_neuron(neuron_input1)
    nn.add_input_neuron(neuron_input2)
    nn.add_hidden_neuron(neuron_hidden1)
    nn.add_hidden_neuron(neuron_hidden2)
    nn.add_output_neuron(neuron_output1)
    nn.add_output_neuron(neuron_output2)

    nn.add_connection(neuron_input1, neuron_hidden1, 1)
    nn.add_connection(neuron_input2, neuron_hidden1, 2)
    nn.add_connection(neuron_input2, neuron_hidden2, 2)
    nn.add_connection(neuron_hidden1, neuron_output1, 4)
    nn.add_connection(neuron_hidden2, neuron_output1, 3)
    nn.add_connection(neuron_input2, neuron_output2, 5)
    nn.add_connection(neuron_hidden1, neuron_output2, 3)

    # hidden_neuron1 = 2*1 + 3*2 = 8
    # hidden_neuron2 = 3*2 = 6
    # output_neuron1 = 8*4 + 6*3 = 50
    # output_neuron2 = 3*5 + 8*3 = 39

    nn.forward_propagation()

    assert neuron_hidden1.value == 8
    assert neuron_hidden2.value == 6
    assert neuron_output1.value == 50
    assert neuron_output2.value == 39

    assert len(nn.connections) == 7
    assert len(nn.hidden_neurons) == 2
    nn.remove_neuron(neuron_hidden2)  # This should remove hidden2 and the connections from/to it
    assert len(nn.connections) == 5
    assert len(nn.hidden_neurons) == 1

    nn.forward_propagation()

    # hidden_neuron1 = 2*1 + 3*2 = 8
    # output_neuron1 = 8*4 = 32
    # output_neuron2 = 3*5 + 8*3 = 39

    assert neuron_hidden1.value == 8
    assert neuron_output1.value == 32
    assert neuron_output2.value == 39


def test_forward_propagation_consistent():
    nn = NeuralNetwork()
    neuron_input1 = Neuron(value=2)
    neuron_hidden1 = Neuron()
    neuron_output1 = Neuron()
    neuron_output2 = Neuron()

    nn.add_input_neuron(neuron_input1)
    nn.add_hidden_neuron(neuron_hidden1)
    nn.add_output_neuron(neuron_output1)
    nn.add_output_neuron(neuron_output2)

    nn.add_connection(neuron_input1, neuron_output1, 3)
    nn.add_connection(neuron_input1, neuron_hidden1, 5)
    nn.add_connection(neuron_hidden1, neuron_output2, 2)

    for i in range(50):
        nn.forward_propagation()

    # output_neuron1 = 2*3 = 6
    # output_neuron2 = 2*5*2 = 20

    assert neuron_output1.value == 6
    assert neuron_output2.value == 20


def test_weight_mutation():
    nn = NeuralNetwork()
    neuron_input1 = Neuron(value=2)
    neuron_input2 = Neuron(value=3)
    neuron_hidden1 = Neuron()
    neuron_hidden2 = Neuron()
    neuron_output1 = Neuron()
    neuron_output2 = Neuron()

    nn.add_input_neuron(neuron_input1)
    nn.add_input_neuron(neuron_input2)
    nn.add_hidden_neuron(neuron_hidden1)
    nn.add_hidden_neuron(neuron_hidden2)
    nn.add_output_neuron(neuron_output1)
    nn.add_output_neuron(neuron_output2)

    nn.add_connection(neuron_input1, neuron_hidden1, 1)
    nn.add_connection(neuron_input2, neuron_hidden1, 2)
    nn.add_connection(neuron_input2, neuron_hidden2, 2)
    nn.add_connection(neuron_hidden1, neuron_output1, 4)
    nn.add_connection(neuron_hidden2, neuron_output1, 3)
    nn.add_connection(neuron_input2, neuron_output2, 5)
    nn.add_connection(neuron_hidden1, neuron_output2, 3)

    mutate_weight(nn.get_connection_between_neurons(neuron_input1, neuron_hidden1), 0.5)

    assert nn.get_connection_between_neurons(neuron_input1, neuron_hidden1).weight == 1.5 or \
           nn.get_connection_between_neurons(neuron_input1, neuron_hidden1).weight == 0.5

    assert nn.get_connection_between_neurons(neuron_input1, neuron_hidden2) is None


def test_random_weight_mutation():
    nn = NeuralNetwork()
    neuron_input1 = Neuron(value=2)
    neuron_output1 = Neuron()
    neuron_output2 = Neuron()

    nn.add_input_neuron(neuron_input1)
    nn.add_output_neuron(neuron_output1)
    nn.add_output_neuron(neuron_output2)

    nn.add_connection(neuron_input1, neuron_output1, 1)
    nn.add_connection(neuron_input1, neuron_output2, 1)

    print("Vor dem mutieren:", nn.forward_propagation().get_output_values())

    nn.random_mutate_weight(0.5)

    print("Nach dem mutieren", nn.forward_propagation().get_output_values())

    first_connection_changed = nn.get_connection_between_neurons(neuron_input1, neuron_output1).weight != 1
    second_connection_changed = nn.get_connection_between_neurons(neuron_input1, neuron_output2).weight != 1

    assert first_connection_changed or second_connection_changed
    assert first_connection_changed != second_connection_changed


def test_set_expected_output_values():
    nn = NeuralNetwork()
    neuron_output1 = OutputNeuron()
    neuron_output2 = OutputNeuron()

    nn.add_output_neuron(neuron_output1)
    nn.add_output_neuron(neuron_output2)

    nn.set_expected_output_values([1, 0])

    assert neuron_output1.expected_value == 1
    assert neuron_output2.expected_value == 0


def test_train():
    nn = NeuralNetwork()
    neuron_input1 = Neuron(value=2)
    neuron_input2 = Neuron(value=3)
    neuron_hidden1 = Neuron()
    neuron_hidden2 = Neuron()
    neuron_output1 = OutputNeuron()
    neuron_output2 = OutputNeuron()

    nn.add_input_neuron(neuron_input1)
    nn.add_input_neuron(neuron_input2)
    nn.add_hidden_neuron(neuron_hidden1)
    nn.add_hidden_neuron(neuron_hidden2)
    nn.add_output_neuron(neuron_output1)
    nn.add_output_neuron(neuron_output2)

    nn.add_connection(neuron_input1, neuron_hidden1, 1)
    nn.add_connection(neuron_input2, neuron_hidden1, 2)
    nn.add_connection(neuron_input2, neuron_hidden2, 2)
    nn.add_connection(neuron_hidden1, neuron_output1, 4)
    nn.add_connection(neuron_hidden2, neuron_output1, 3)
    nn.add_connection(neuron_input2, neuron_output2, 5)
    nn.add_connection(neuron_hidden1, neuron_output2, 3)

    nn.set_expected_output_values([1, 0])

    nn.train(10000, 0.1)
    nn.forward_propagation()

    delta_shift = 0.1
    output1_difference = abs(nn.output_neurons[0].value - nn.output_neurons[0].expected_value)
    output2_difference = abs(nn.output_neurons[1].value - nn.output_neurons[1].expected_value)

    # TODO: Test is not consistent
    # Training works but it's a little bit random
    # We should update training to set a minimum fitness or something like that

    assert output1_difference < delta_shift
    assert output2_difference < delta_shift

# TODO: test copy
# TODO: test get_fitness
