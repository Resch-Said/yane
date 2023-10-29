from src.neural_network.ActivationFunction import ActivationFunction
from src.neural_network.YaneConfig import get_fire_rate_min


class Neuron:
    def __init__(self, value=0.0, fire_rate=get_fire_rate_min()):
        self.value = value
        self.fire_rate_fixed = fire_rate
        self.fire_rate_variable = self.fire_rate_fixed
        self.activation_function = ActivationFunction.SIGMOID
