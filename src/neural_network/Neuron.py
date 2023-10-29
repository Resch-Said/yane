import enum
import random

from src.neural_network.YaneConfig import get_fire_rate_min, get_fire_rate_max, get_random_activation_function


class Neuron:
    def __init__(self, value=0.0, fire_rate=None, activation_function=None):

        if fire_rate is None:
            fire_rate = random.randint(get_fire_rate_min(), get_fire_rate_max())

        if activation_function is None:
            activation_function = get_random_activation_function()

        self.value = value
        self.fire_rate_fixed = fire_rate
        self.fire_rate_variable = fire_rate
        self.activation_function = activation_function
