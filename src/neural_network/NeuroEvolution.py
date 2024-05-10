import numpy as np
from line_profiler_pycharm import profile

from src.neural_network.Population import Population
from src.neural_network.Species import Species
from src.neural_network.genome.Genome import Genome
from src.neural_network.util import YaneConfig

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

    @profile
    def train(self, callback_evaluation):
        if self.get_genomes_size() > 0:
            for genome in self.get_population().get_all_genomes():
                self.add_evaluation(genome)

            self.clear_population()

        while True:
            self.clear_bad_species_genomes()
            self.create_next_genomes()
            self.evaluate_next_genome(callback_evaluation)

            print("Generation: " + str(np.round(self.get_generation())) + " Best fitness: " + str(
                self.get_best_fitness()) + " Average fitness: " + str(self.get_average_fitness()),
                  "Number of species: " + str(self.get_population().get_species_size()))

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

    @profile
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

    @profile
    def evaluate_next_genome(self, callback_evaluation):
        while len(self.get_evaluation_list()) > 0:
            genome = self.get_evaluation_list().pop()
            genome.evaluate(callback_evaluation)
            self.add_population(genome)
            self.generation += 1 / YaneConfig.get_max_population_size(yane_config)

    def add_evaluation(self, genome):
        if isinstance(genome, list):
            self.evaluation_list.extend(genome)
        else:
            self.evaluation_list.append(genome)

    def get_evaluation_list(self):
        return self.evaluation_list

    def create_next_genomes(self):
        if self.get_genomes_size() <= 0:
            return

        random_species = self.get_population().get_random_species()

        # TODO: Add different crossover methods

        child_genomes: list[Genome] = [genome.copy() for genome in random_species.get_upper_genomes()]

        for child, parent in zip(child_genomes, random_species.get_upper_genomes()):
            child.set_parent(parent)
            child.mutate()
            parent.set_reproduction_count(parent.get_reproduction_count() + 1)

        self.add_evaluation(child_genomes)

    def get_best_species_genome(self) -> (Species, Genome):
        return self.get_population().get_best_species_genome()

    def remove_species(self, species):
        self.get_population().remove_species(species)

    def clear_overpopulated_species(self):
        for species in self.get_population().get_species():
            while species.get_size() > YaneConfig.get_species_size_reference(yane_config):
                species.pop_genome()

    def clear_stagnated_species(self, species: Species):

        top_genomes = self.get_population().get_top_genomes(YaneConfig.get_elitism(yane_config))

        if species.get_generations_without_improvement() > YaneConfig.get_species_stagnation_duration(yane_config):
            for genome in species.get_genomes():
                if genome in top_genomes:
                    self.add_evaluation(species.get_best_genome())
            self.remove_species(species)

    def set_min_fitness(self, min_fitness):
        self.min_fitness = min_fitness

    def clear_bad_reproducers(self, species: Species):
        top_genomes = self.get_population().get_top_genomes(YaneConfig.get_elitism(yane_config))

        for genome in species.get_genomes()[:]:
            if genome.get_bad_reproduction_count() > YaneConfig.get_max_bad_reproductions_in_row(
                    yane_config) and genome not in top_genomes:
                species.remove_genome(genome)

    def clear_population(self):
        self.get_population().clear()

    def clear_bad_species_genomes(self):
        for species in self.get_population().get_species()[:]:
            self.clear_stagnated_species(species)
            self.clear_bad_reproducers(species)
            self.clear_empty_species(species)

    def clear_empty_species(self, species):
        if species.get_size() <= 0:
            self.remove_species(species)
