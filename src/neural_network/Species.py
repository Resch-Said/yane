import bisect
import random

import numpy as np

from src.neural_network import YaneConfig
from src.neural_network.Genome import Genome

yane_config = YaneConfig.load_json_config()


class Species:

    def __init__(self):
        self.genomes: list[Genome] = []
        self.average_fitness = None
        self.previous_average_fitness = None
        self.generations_without_improvement = 0

    def get_generations_without_improvement(self):
        return self.generations_without_improvement

    def add_genome(self, genome):
        bisect.insort(self.genomes, genome, key=lambda x: -x.get_fitness())

        if self.average_fitness is None:
            self.average_fitness = self.get_average_fitness()
            self.previous_average_fitness = self.average_fitness

        self.prune_overpopulation()
        self.update_average_fitness()
        self.update_generations_without_improvement()

    def pop_genome(self):
        self.genomes.pop()

    def remove_genome(self, genome):
        self.genomes.remove(genome)

    def get_genomes(self):
        return self.genomes

    def get_size(self):
        return len(self.genomes)

    def get_average_fitness(self):
        if self.get_size() <= 0:
            return None

        sum_fitness = 0
        for genome in self.genomes:
            sum_fitness += genome.get_fitness()
        return sum_fitness / self.get_size()

    def get_best_fitness(self):
        if self.get_size() <= 0:
            return None

        return self.genomes[0].get_fitness()

    def get_best_genome(self) -> Genome | None:
        if self.get_size() <= 0:
            return None

        return self.genomes[0]

    def print(self):
        print("Species size: " + str(self.get_size()))
        print("Average fitness: " + str(self.get_average_fitness()))
        print("Best fitness: " + str(self.get_best_fitness()))
        print("Generations without improvement: " + str(self.get_generations_without_improvement()))

    # smaller is better
    def get_species_compatibility(self, genome):
        if self.get_size() <= 0:
            return None

        return self.genomes[0].get_species_compatibility(genome)

    def update_average_fitness(self):
        new_average_fitness = self.get_average_fitness()

        if new_average_fitness > self.average_fitness:
            self.average_fitness = new_average_fitness

    def update_generations_without_improvement(self):

        improved_percentage = (self.average_fitness - self.previous_average_fitness) / self.previous_average_fitness

        if improved_percentage >= YaneConfig.get_improvement_threshold(yane_config):
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1 / YaneConfig.get_species_size_reference(yane_config)

    def get_random_genome(self) -> Genome:
        fraction = YaneConfig.get_reproduction_fraction(yane_config)

        reproduction_limit = int(np.ceil(fraction * self.get_size()))

        if reproduction_limit <= 0:
            reproduction_limit = 1

        return random.choice(self.genomes[:reproduction_limit])

    def prune_overpopulation(self):
        while self.get_size() > YaneConfig.get_species_size_reference(yane_config):
            self.pop_genome()
