from src.neural_network.ActivationFunction import ActivationFunction
from src.neural_network.Neuron import Neuron
from src.neural_network.exceptions.InvalidActivation import InvalidActivation
from src.neural_network.exceptions.InvalidMutation import InvalidMutation


class InputNeuron(Neuron):
    def __init__(self):
        super().__init__()
        self.activation = ActivationFunction.LINEAR

    def fire(self):
        for connection in self.next_connections:
            if not connection.is_enabled():
                continue

            next_neuron: Neuron = connection.get_out_neuron()
            next_neuron.set_value(next_neuron.get_value() + self.value * connection.get_weight())

    def mutate_activation_function(self):
        raise InvalidMutation("Cannot mutate activation function of input neuron")

    def set_activation(self, activation):
        raise InvalidActivation("Cannot set activation function of input neuron")
