import bisect

from src.neural_network.Genome import Genome


class Population:

    def __init__(self):
        self.genomes: list[Genome] = []

    def add_genome(self, genome):
        bisect.insort_left(self.genomes, genome, key=lambda x: -x.get_fitness())

    # TODO: make this more smarter so genomes have a chance to survive.
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

        total = 0
        for genome in self.genomes:
            total += genome.get_fitness()
        return total / self.get_size()

    def print_fitness(self):
        print("Population:")
        for genome in self.genomes:
            print(genome.get_fitness())
