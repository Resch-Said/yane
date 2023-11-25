import random
from copy import deepcopy

from src.neural_network import YaneConfig, Neuron

yane_config = YaneConfig.load_json_config()


class Connection:
    ID = 0

    def __init__(self, in_neuron: Neuron = None, out_neuron: Neuron = None, weight: float = 1.0, enabled: bool = True):
        self.weight = weight
        self.in_neuron = in_neuron
        self.out_neuron = out_neuron
        self.enabled = enabled
        self.id = Connection.ID
        Connection.ID += 1

    def set_weight(self, weight):
        self.weight = weight

    def set_in_neuron(self, neuron):
        self.in_neuron = neuron

    def set_out_neuron(self, neuron):
        self.out_neuron = neuron

    def set_enabled(self, enabled):
        self.enabled = enabled

    def get_weight(self):
        return self.weight

    def get_in_neuron(self) -> Neuron:
        return self.in_neuron

    def get_out_neuron(self) -> Neuron:
        return self.out_neuron

    def is_enabled(self) -> bool:
        return self.enabled

    def __str__(self):
        return "Connection: " + str(self.id) + " from " + str(self.in_neuron.id) + " to " + str(
            self.out_neuron.id) + " with weight " + str(self.weight) + " and enabled: " + str(self.enabled)

    def get_id(self) -> int:
        return self.id

    def copy(self) -> 'Connection':
        return deepcopy(self)

    def mutate_weight_random(self):
        self.weight = YaneConfig.get_random_mutation_weight(yane_config)

    def mutate_enabled(self):
        self.enabled = not self.enabled

    def mutate_weight_shift(self):
        if random.random() < 0.5:
            self.weight += YaneConfig.get_random_weight_shift(yane_config)
        else:
            self.weight -= YaneConfig.get_random_weight_shift(yane_config)

        if self.weight < YaneConfig.get_mutation_weight_min(yane_config):
            self.weight = YaneConfig.get_mutation_weight_min(yane_config)
        elif self.weight > YaneConfig.get_mutation_weight_max(yane_config):
            self.weight = YaneConfig.get_mutation_weight_max(yane_config)
