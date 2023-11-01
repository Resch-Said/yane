# TODO:Manage the neural networks.
# It should be able to create, train, and test the neural networks.
# Kill 50% of the networks that are not performing well.
# Kill last 40% of the networks that are not performing well randomly with a 10% chance of survival.
# The Top 10% of the networks should be cloned and mutated.
# To fill the population, the remaining networks are randomly populated.
import cProfile
import random
from copy import deepcopy

from src.neural_network.NeuralNetwork import NeuralNetwork
from src.neural_network.YaneConfig import get_population_size, get_population_survival_rate, load_json_config

json_config = load_json_config()


class NeuroCluster:
    def __init__(self):
        self.neural_networks = []

        for i in range(get_population_size(json_config)):
            nn = NeuralNetwork()
            self.neural_networks.append(nn)

    def set_custom_fitness(self, custom_fitness):
        for nn in self.neural_networks:
            nn.custom_fitness = custom_fitness

    # optimize the neural networks.
    # sort the neural networks by fitness.
    # kill the under 50% of the neural networks.
    # kill the last 40% of the neural networks randomly with a 10% chance of survival.
    # clone the rest of the neural networks and mutate them until the population is full.
    # repeat until the wished amount of generations is reached or the fitness is good enough.

    def train_population(self, min_fitness=-0.1, max_iterations=1000, fitness_tolerance=0.01):

        self.optimize_population(fitness_tolerance)

        while self.neural_networks[0].fitness < min_fitness and max_iterations > 0:
            max_iterations -= 1

            self.optimize_population(fitness_tolerance)

            self.neural_networks.sort(key=lambda x: x.fitness, reverse=True)

            self.thin_out_population()
            self.regrow_population()

            print("Best current fitness: " + str(self.neural_networks[0].fitness))

    def optimize_population(self, fitness_tolerance):
        for nn in self.neural_networks:
            nn.optimize_weights(fitness_tolerance)

    def regrow_population(self):
        while len(self.neural_networks) < get_population_size(json_config):
            nn_parent = random.choice(self.neural_networks)
            nn_child = deepcopy(nn_parent)
            nn_child.mutate()
            self.neural_networks.append(nn_child)

    def thin_out_population(self):

        for i in range(int(len(self.neural_networks) * get_population_survival_rate(json_config))):
            self.neural_networks.pop()
            if random.random() < 0.5:
                random_nn = random.choice(self.neural_networks)
                random_nn2 = random.choice(self.neural_networks)
                if random_nn.fitness > random_nn2.fitness:
                    self.neural_networks.remove(random_nn2)
                else:
                    self.neural_networks.remove(random_nn)

    def print(self):
        pass

    def set_expected_output_values(self, param):
        for nn in self.neural_networks:
            nn.set_expected_output_values(param)

    def set_input_neurons(self, param):
        for nn in self.neural_networks:
            nn.set_input_neurons(param)


def main():
    nc = NeuroCluster()

    nc.set_input_neurons([5, 10])
    nc.set_expected_output_values(list(range(10)))

    nc.train_population()

    nc.neural_networks[0].print()


if __name__ == '__main__':
    cProfile.run('main()', sort='cumtime')
