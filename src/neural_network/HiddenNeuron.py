from src.neural_network.Neuron import Neuron


class HiddenNeuron(Neuron):
    def __init__(self, value=0.0, fire_rate=None, activation_function=None):
        super().__init__(value=value, fire_rate=fire_rate, activation_function=activation_function)
