import random

from src.neural_network import YaneConfig, Genome
from src.neural_network.Species import Species

yane_config = YaneConfig.load_json_config()


class Population:
    def __init__(self):
        self.species: list[Species] = []
        self.compatibility_threshold = 3
        self.last_checked_species_size = 0

    def get_species(self):
        return self.species

    def get_random_species(self) -> Species:
        return random.choice(self.species)

    def get_genomes_size(self):
        population_size = 0
        for species in self.species:
            population_size += species.get_size()

        return population_size

    def get_species_size(self):
        return len(self.species)

    def get_average_fitness(self):
        if self.get_genomes_size() <= 0:
            return None

        sum_fitness = 0
        for species in self.species:
            sum_fitness += species.get_average_fitness()

        return sum_fitness / self.get_species_size()

    def get_best_fitness(self):
        if self.get_genomes_size() <= 0:
            return None

        return self.get_best_species_genome()[1].get_fitness()

    def print(self):
        print("Population size: " + str(self.get_genomes_size()))
        print("Average fitness: " + str(self.get_average_fitness()))
        print("Best fitness: " + str(self.get_best_fitness()))

    def get_best_species_genome(self) -> (Species, Genome):
        if self.get_genomes_size() <= 0:
            return None

        best_genome = None
        best_species = None

        for species in self.species:
            if best_genome is None or species.get_best_fitness() > best_genome.get_fitness():
                best_genome = species.get_best_genome()
                best_species = species

        return best_species, best_genome

    def add_genome(self, genome):
        if self.get_species_size() <= 0:
            species = Species()
            species.add_genome(genome)
            self.species.append(species)
            return

        species, compatibility = self.get_best_compatible_species(genome)
        has_species_size_increased = self.last_checked_species_size < self.get_species_size()
        has_species_size_decreased = self.last_checked_species_size > self.get_species_size()
        self.last_checked_species_size = self.get_species_size()

        if self.get_species_size() > YaneConfig.get_max_species_per_population(
                yane_config) and has_species_size_increased:
            self.compatibility_threshold *= 1.5  # reduces the number of species
        elif self.get_species_size() < YaneConfig.get_max_species_per_population(
                yane_config) and has_species_size_decreased:
            self.compatibility_threshold /= 1.5  # increases the number of species

        if compatibility <= self.compatibility_threshold:
            species.add_genome(genome)
            return

        species = Species()
        species.add_genome(genome)
        self.species.append(species)

    def get_best_compatible_species(self, genome) -> (Species, float):
        best_species_compatibility = None
        best_species = None
        for species in self.species:
            species_compatibility = species.get_species_compatibility(genome)
            if best_species_compatibility is None or species_compatibility < best_species_compatibility:
                best_species_compatibility = species_compatibility
                best_species = species

        return best_species, best_species_compatibility

    def get_all_genomes(self):
        genomes = []
        for species in self.species:
            genomes += species.get_genomes()

        return genomes

    def remove_species(self, species):
        self.species.remove(species)
