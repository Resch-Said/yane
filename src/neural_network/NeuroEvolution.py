import random
from multiprocessing import Process
from time import sleep

from src.neural_network import YaneConfig
from src.neural_network.Genome import Genome
from src.neural_network.Population import Population

yane_config = YaneConfig.load_json_config()


class NeuroEvolution:
    def __init__(self, output_neurons=1):
        self.population = Population(output_neurons=output_neurons)
        self.evaluation_list = []
        self.ready_for_population_list = []
        self.finished = False

    @classmethod
    def crossover(cls, genome1, genome2) -> Genome:
        return Genome.crossover(genome1, genome2)

    def get_population(self):
        return self.population

    def get_evaluation_list(self):
        return self.evaluation_list

    def get_ready_for_population_list(self):
        return self.ready_for_population_list

    def add_evaluation(self, genome):
        self.evaluation_list.append(genome)

    def add_ready_for_population(self, genome):
        self.ready_for_population_list.append(genome)

    def remove_evaluation(self, genome):
        self.evaluation_list.remove(genome)

    def remove_ready_for_population(self, genome):
        self.ready_for_population_list.remove(genome)

    def add_genome_evaluation(self, genome):
        self.evaluation_list.append(genome)

    def pop_genome(self):
        self.population.pop_genome()

    def remove_genome(self, genome):
        self.population.remove_genome(genome)

    def get_genomes(self):
        return self.population.get_genomes()

    def get_size(self):
        return self.population.get_size()

    def add_output_neuron(self, neuron):
        self.population.add_output_neuron(neuron)

    # TODO: implement this method
    def train(self, min_fitness=None):
        p_population_limiter = Process(target=self.reduce_overpopulation_multiprocessing)
        p_population_breeder = Process(target=self.breed_population_multiprocessing)

    # TODO: implement this method
    def breed_population_multiprocessing(self):

        while not self.finished:
            genome1 = self.get_random_genome()
            genome2 = self.get_random_genome()

            child_genome = self.crossover(genome1, genome2)
            child_genome.mutate()

    def reduce_overpopulation_multiprocessing(self):
        timer = 0

        while not self.finished:
            if self.get_size() > YaneConfig.get_population_size(yane_config):
                self.pop_genome()
                timer = 0
            else:
                sleep(timer)
                timer += 1

    def get_random_genome(self):
        return random.choice(self.get_genomes())
