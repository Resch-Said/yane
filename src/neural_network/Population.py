from bisect import insort

from src.neural_network.Genome import Genome
from src.neural_network.OutputNeuron import OutputNeuron


class Population:

    def __init__(self):
        self.genomes = []

    def add_genome(self, genome):
        insort(self.genomes, genome, key=lambda x: x.get_fitness())

    def pop_genome(self):
        self.genomes.pop()

    def remove_genome(self, genome):
        self.genomes.remove(genome)

    def get_genomes(self):
        return self.genomes

    def get_size(self):
        return len(self.genomes)

    def sort(self):
        self.genomes.sort(key=lambda x: x.get_fitness(), reverse=True)

    def get_average_fitness(self):
        total = 0
        for genome in self.genomes:
            total += genome.get_fitness()
        return total / len(self.genomes)

    def add_output_neuron(self, neuron: OutputNeuron):
        if len(self.genomes) <= 0:
            new_genome = Genome()
            self.add_genome(new_genome)

        genome: Genome
        for genome in self.genomes:
            genome.add_output_neuron(neuron)
