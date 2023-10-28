from src.neural_network.Neuron import Neuron


class InputNeuron(Neuron):
    def __init__(self, value=0.0):
        super().__init__()
        self.value = value
