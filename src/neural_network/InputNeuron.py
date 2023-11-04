from src.neural_network.ActivationFunction import ActivationFunction
from src.neural_network.Neuron import Neuron


class InputNeuron(Neuron):
    def __init__(self):
        super().__init__()
        self.activation = ActivationFunction.LINEAR

    def fire(self):
        for connection in self.next_connections:
            next_neuron: Neuron = connection.get_out_neuron()
            next_neuron.set_value(next_neuron.get_value() + self.value * connection.get_weight())
