from src.neural_network.Neuron import Neuron


class OutputNeuron(Neuron):
    def __init__(self, value=0):
        super().__init__()
        self.value = value
        self.expected_value = 0.0
