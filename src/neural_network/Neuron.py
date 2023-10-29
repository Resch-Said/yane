from src.neural_network.ActivationFunction import ActivationFunction
from src.neural_network.YaneConfig import get_fire_rate_min


class Neuron:
    def __init__(self, value=0.0):
        self.value = value
        self.fire_rate_fixed = get_fire_rate_min()
        self.fire_rate_variable = self.fire_rate_fixed
        self.activation_function = ActivationFunction.SIGMOID
