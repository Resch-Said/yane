import random
from copy import deepcopy

import numpy as np

from src.neural_network import YaneConfig
from src.neural_network.Genome import Genome
from src.neural_network.Population import Population

yane_config = YaneConfig.load_json_config()


class NeuroEvolution:
    def __init__(self):
        self.max_iterations = None
        self.current_generation = 0
        self.min_fitness = None
        self.population = Population()
        self.evaluation_list: list[Genome] = []

    @classmethod
    def crossover(cls, genome1, genome2) -> Genome:
        new_genome = Genome.crossover(genome1, genome2)

        return new_genome

    def get_population(self):
        return self.population

    def pop_genome(self):
        self.population.pop_genome()

    def remove_genome(self, genome):
        self.population.remove_genome(genome)

    def get_genomes_population(self):
        return self.population.get_genomes()

    def get_size(self):
        return self.population.get_size()

    # TODO: implement this method
    def train(self, callback_evaluation):
        while True:
            self.current_generation += 1 / YaneConfig.get_population_size(yane_config)

            self.evaluate_next_genome(callback_evaluation)
            self.create_next_genome()

            if self.get_size() > YaneConfig.get_population_size(yane_config):
                self.pop_genome()  # TODO: New genomes have a hard time to not be removed

            print("Generation: " + str(np.round(self.current_generation)) + " Best fitness: " + str(
                self.get_best_fitness()), end='\r')

            if self.check_best_fitness() or self.check_max_iterations():
                break

    def get_random_genome(self):

        weights = []
        for i in range(self.get_size()):
            weights.append(self.get_size() - i)

        return random.choices(self.get_genomes_population(), weights=weights)[0]

    def print(self):
        print("Population size: " + str(self.get_size()))
        print("Average fitness: " + str(self.get_average_fitness()))
        print("Best fitness: " + str(self.get_best_fitness()))

    def get_average_fitness(self):
        return self.population.get_average_fitness()

    def get_best_fitness(self):
        return self.get_genomes_population()[0].get_fitness()

    def add_population(self, genome: Genome):
        self.population.add_genome(genome)

    def set_max_generations(self, iterations):
        self.max_iterations = iterations

    def check_best_fitness(self):
        if self.min_fitness is None:
            return

        if self.get_best_fitness() >= self.min_fitness:
            return True

        return False

    def check_max_iterations(self):
        if self.max_iterations is None:
            return

        if self.current_generation >= self.max_iterations:
            return True

        return False

    def set_number_of_outputs(self, number_of_outputs):
        if self.get_size() <= 0:
            genome = Genome()
            genome.set_number_of_outputs(number_of_outputs)
            self.add_evaluation(genome)
        else:
            for genome in self.get_genomes_population():
                genome.set_number_of_outputs(number_of_outputs)

    def evaluate_next_genome(self, callback_evaluation):
        if len(self.get_evaluation_list()) <= 0:
            return

        genome = self.get_evaluation_list().pop()
        genome.evaluate(callback_evaluation)
        self.add_population(genome)

    def add_evaluation(self, genome):
        self.evaluation_list.append(genome)

    def get_evaluation_list(self):
        return self.evaluation_list

    def create_next_genome(self):
        if self.get_size() <= 0:
            return

        genome1 = self.get_random_genome()
        genome2 = self.get_random_genome()

        # child_genome = self.crossover(genome1, genome2)
        child_genome: Genome = deepcopy(genome1)
        child_genome.reset_forward_order()

        child_genome.mutate()

        self.add_evaluation(child_genome)
