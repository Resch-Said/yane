import random
import numpy as np

from neural_network.mutation import Mutation


class Connection:
    def __init__(self, neuron) -> None:
        self.neuron = neuron
        self.weight = 0
        self.mutation = Mutation()

    def mutate(self) -> None:
        self.weight = self.mutation.mutate_value(self.weight)
        self.mutation.mutate_rates()
        
    def copy(self):
        connection = Connection(self.neuron.copy())
        connection.weight = self.weight
        connection.mutation = self.mutation.copy()
        return connection