from src.neural_network.Neuron import Neuron


class Connection:
    def __init__(self, neuron_from: Neuron, neuron_to: Neuron, weight=1.0):
        self.neuron_from = neuron_from
        self.neuron_to = neuron_to
        self.weight = weight
