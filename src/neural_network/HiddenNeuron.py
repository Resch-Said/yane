from src.neural_network.Neuron import Neuron


class HiddenNeuron(Neuron):
    def __init__(self, value=0):
        super().__init__()
        self.value = value
