from src.neural_network.ActivationFunction import ActivationFunction
from src.neural_network.Neuron import Neuron


class InputNeuron(Neuron):
    def __init__(self):
        super().__init__()
        self.activation = ActivationFunction.LINEAR
