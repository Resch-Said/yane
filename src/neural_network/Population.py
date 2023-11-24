from bisect import insort

from src.neural_network.Genome import Genome
from src.neural_network.OutputNeuron import OutputNeuron


class Population:

    def __init__(self, output_neurons=1):
        self.genomes = []
        for i in range(output_neurons):
            new_output_neuron: OutputNeuron = OutputNeuron()
            self.add_output_neuron(new_output_neuron)

    def add_genome(self, genome):
        insort(self.genomes, genome, key=lambda x: x.get_fitness())
        self.sort()

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

    # TODO: find better way to initialize population
    def add_output_neuron(self, neuron: OutputNeuron):
        if len(self.genomes) <= 0:
            new_genome = Genome()
            new_genome.add_output_neuron(neuron)
            new_genome.evaluate()
            self.add_genome(new_genome)
        else:
            for genome in self.genomes:
                genome.add_output_neuron(neuron)
