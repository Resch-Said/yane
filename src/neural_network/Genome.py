from copy import deepcopy

from src.neural_network.NeuralNetwork import NeuralNetwork


class Genome:
    def __init__(self):
        self.brain = NeuralNetwork()
        self.fitness = 0.0
        self.net_cost = 0.0

    def get_brain(self):
        return self.brain

    def get_fitness(self):
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_net_cost(self):
        return self.net_cost

    def set_net_cost(self, net_cost):
        self.net_cost = net_cost

    def evaluate(self):
        self.set_fitness(self.get_brain().evaluate())
        self.set_net_cost(self.get_brain().get_net_cost())
        return self.get_fitness()

    def copy(self):
        return deepcopy(self)
