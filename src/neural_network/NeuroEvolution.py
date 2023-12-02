import numpy as np

from src.neural_network import YaneConfig
from src.neural_network.Genome import Genome
from src.neural_network.Population import Population
from src.neural_network.Species import Species

yane_config = YaneConfig.load_json_config()


class NeuroEvolution:
    def __init__(self):
        self.max_generations = None
        self.min_fitness = None
        self.generation = 0
        self.population = Population()
        self.evaluation_list: list[Genome] = []

    def get_population(self):
        return self.population

    def get_genomes_size(self):
        return self.population.get_genomes_size()

    def get_generation(self):
        return self.generation

    def train(self, callback_evaluation):
        while True:
            current_generation = self.get_generation()

            self.evaluate_next_genome(callback_evaluation)
            self.create_next_genome()

            self.generation += 1 / YaneConfig.get_max_population_size(yane_config)

            overpopulation_count = self.get_genomes_size() - YaneConfig.get_max_population_size(yane_config)

            print("Generation: " + str(np.round(current_generation)) + " Best fitness: " + str(
                self.get_best_fitness()) + " Average fitness: " + str(self.get_average_fitness()),
                  "Number of species: " + str(self.get_population().get_species_size()))

            if overpopulation_count > 0:
                self.clear_stagnated_species()
                self.clear_overpopulated_species()
                # self.clear_bad_genomes()
                self.clear_bad_reproducers()

            if self.check_best_fitness() or self.check_max_generation():
                break

    def print(self):
        print("Population size: " + str(self.get_genomes_size()))
        print("Number of species: " + str(self.get_population().get_species_size()))
        print("Average fitness: " + str(self.get_average_fitness()))
        print("Best fitness: " + str(self.get_best_fitness()))

    def get_average_fitness(self):
        return self.population.get_average_fitness()

    def get_best_fitness(self):
        return self.population.get_best_fitness()

    def add_population(self, genome: Genome):
        self.get_population().add_genome(genome)

    def set_max_generations(self, generations):
        self.max_generations = generations

    def check_best_fitness(self):
        if self.min_fitness is None:
            return

        best_fitness = self.get_best_fitness()

        if best_fitness is None:
            return

        if best_fitness >= self.min_fitness:
            return True

        return False

    def check_max_generation(self):
        if self.max_generations is None:
            return

        if self.generation >= self.max_generations:
            return True

        return False

    def set_number_of_outputs(self, number_of_outputs):
        if self.get_genomes_size() <= 0:
            genome = Genome()
            genome.set_number_of_outputs(number_of_outputs)
            self.add_evaluation(genome)
        else:
            for genome in self.get_population().get_all_genomes():
                genome.set_number_of_outputs(number_of_outputs)

    def evaluate_next_genome(self, callback_evaluation):
        while len(self.get_evaluation_list()) > 0:
            genome = self.get_evaluation_list().pop()
            genome.evaluate(callback_evaluation)
            self.add_population(genome)

    def add_evaluation(self, genome):
        self.evaluation_list.append(genome)

    def get_evaluation_list(self):
        return self.evaluation_list

    def create_next_genome(self):
        if self.get_genomes_size() <= 0:
            return

        random_species = self.get_population().get_random_species()

        # TODO: Add different crossover methods

        genome1 = random_species.get_random_genome()
        # genome2 = random_species.get_random_genome()

        # child_genome = Genome.crossover(genome1, genome2)
        child_genome: Genome = genome1.copy()

        # child_genome.set_best_parent_fitness(max(genome1.get_fitness(), genome2.get_fitness()))
        child_genome.set_parent(genome1)
        child_genome.mutate()
        # child_genome.prune_bad_connections()

        genome1.set_reproduction_count(genome1.get_reproduction_count() + 1)
        # genome2.set_reproduction_count(genome2.get_reproduction_count() + 1)

        self.add_evaluation(child_genome)

    def get_best_species_genome(self) -> (Species, Genome):
        return self.get_population().get_best_species_genome()

    def remove_species(self, species):
        self.get_population().remove_species(species)

    def clear_overpopulated_species(self):
        for species in self.get_population().get_species():
            while species.get_size() > YaneConfig.get_species_size_reference(yane_config):
                species.pop_genome()

    def clear_stagnated_species(self):

        best_genome = self.get_best_species_genome()[1]

        for species in self.get_population().get_species():
            if species.get_generations_without_improvement() > YaneConfig.get_species_stagnation_duration(yane_config):
                if species.get_best_genome() is not best_genome:
                    self.remove_species(species)

    def set_min_fitness(self, min_fitness):
        self.min_fitness = min_fitness

    def clear_bad_reproducers(self):
        for species in self.get_population().get_species():
            for genome in species.get_genomes():
                if genome.get_bad_reproduction_count() > YaneConfig.get_max_bad_reproductions_in_row(yane_config):
                    if genome is not species.get_best_genome():
                        species.remove_genome(genome)

    def clear_bad_genomes(self):
        for species in self.get_population().get_species():
            for genome in species.get_genomes():
                if genome.get_fitness() < species.get_best_genome().get_fitness() * 1:
                    species.remove_genome(genome)
