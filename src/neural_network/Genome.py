from copy import deepcopy
from random import random

import numpy as np

from src.neural_network import YaneConfig
from src.neural_network.HiddenNeuron import HiddenNeuron
from src.neural_network.NeuralNetwork import NeuralNetwork
from src.neural_network.Neuron import Neuron
from src.neural_network.OutputNeuron import OutputNeuron

yane_config = YaneConfig.load_json_config()


class Genome:
    def __init__(self, neuron_genes=None):
        self.bad_reproduction_count = 0
        self.brain: NeuralNetwork = NeuralNetwork()
        self.parent: Genome | None = None
        self.fitness = None
        self.net_cost = None
        self.reproduction_count = 0

        if neuron_genes is not None:
            for neuron in neuron_genes:
                self.add_neuron(neuron)

    @classmethod
    def crossover_connections(cls, genome1, genome2):
        connection_genes1 = genome1.get_brain().get_all_connections()
        connection_genes2 = genome2.get_brain().get_all_connections()

        return Genome.crossover_genes(connection_genes1, connection_genes2)

    @classmethod
    def align_gene_ids(cls, genes1, genes2):
        aligned_genes = []

        index1 = 0
        index2 = 0

        while index1 < len(genes1) and index2 < len(genes2):
            neuron1 = genes1[index1]
            neuron2 = genes2[index2]

            if neuron1.get_id() == neuron2.get_id():
                aligned_genes.append((neuron1, neuron2))
                index1 += 1
                index2 += 1
            elif neuron1.get_id() < neuron2.get_id():
                aligned_genes.append((neuron1, None))
                index1 += 1
            elif neuron1.get_id() > neuron2.get_id():
                aligned_genes.append((None, neuron2))
                index2 += 1

        while index1 < len(genes1):
            aligned_genes.append((genes1[index1], None))
            index1 += 1

        while index2 < len(genes2):
            aligned_genes.append((None, genes2[index2]))
            index2 += 1

        return aligned_genes

    @classmethod
    def crossover_genes(cls, gene1, gene2) -> list:
        aligned_genes = Genome.align_gene_ids(gene1, gene2)

        new_genes = []

        for gene1, gene2 in aligned_genes:
            if gene1 is None:
                new_genes.append(gene2)
            elif gene2 is None:
                new_genes.append(gene1)
            else:
                if random() < 0.5:
                    new_genes.append(gene1)
                else:
                    new_genes.append(gene2)

        return deepcopy(new_genes)

    @classmethod
    def crossover_neurons(cls, genome1, genome2) -> list:
        neuron_genes1 = genome1.get_brain().get_all_neurons()
        neuron_genes2 = genome2.get_brain().get_all_neurons()

        return Genome.crossover_genes(neuron_genes1, neuron_genes2)

    @classmethod
    def crossover(cls, genome1, genome2) -> 'Genome':
        neuron_genes = Genome.crossover_neurons(genome1, genome2)
        child_genome = Genome(neuron_genes)

        return child_genome

    def get_brain(self):
        return self.brain

    def get_fitness(self):
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_net_cost(self):
        return self.net_cost

    def set_net_cost(self, net_cost):
        self.net_cost = net_cost

    # callback_evaluator is a function that takes a genome as a parameter and returns a fitness value
    # This function is used to evaluate the fitness of a genome
    # You have to implement this function yourself since it is specific to your problem
    def evaluate(self, callback_evaluator):
        self.set_net_cost(self.get_brain().calculate_net_cost())

        fitness_result = callback_evaluator(self)

        self.clear_hidden_output_neurons()

        net_cost = self.get_net_cost()

        if self.parent is not None and fitness_result >= self.parent.get_fitness():
            self.parent.set_bad_reproduction_count(0)

        # Child genome is worse than parent genome
        if self.parent is not None and fitness_result < self.parent.get_fitness():
            parent_connection = self.get_parent().get_brain().get_last_weight_shift_connection()
            self.parent.set_bad_reproduction_count(self.parent.get_bad_reproduction_count() + 1)
            if parent_connection is not None:
                parent_connection.switch_weight_shift_direction()

        # TODO: Remove net cost as soon as fitness prioritization is implemented
        # self.set_fitness(fitness_result - net_cost * YaneConfig.get_net_cost_factor(yane_config))
        self.set_fitness(fitness_result)
        return self.get_fitness()

    def copy(self):
        return deepcopy(self)

    def add_output_neuron(self, neuron: OutputNeuron):
        self.brain.add_output_neuron(neuron)

    def add_hidden_neuron(self, neuron: HiddenNeuron):
        self.brain.add_hidden_neuron(neuron)

    def add_neuron(self, neuron: Neuron):
        self.brain.add_neuron(neuron)

    def remove_all_connections(self):
        self.brain.remove_all_connections()

    def mutate(self):
        self.brain.mutate()

    def print(self):
        print("Genome: " + str(self.get_fitness()) + " with net cost: " + str(self.get_net_cost()) + " and " + str(
            len(self.get_brain().get_forward_order_list())) + " connected neurons")
        self.brain.print()

    def get_all_neurons(self):
        return self.brain.get_all_neurons()

    def add_connection(self, connection):
        self.brain.add_connection(connection)

    def set_input_data(self, data):
        self.brain.set_input_data(data)

    def forward_propagation(self, data=None):
        self.brain.forward_propagation(data)

    def get_outputs(self) -> list:
        return self.brain.get_output_data()

    def set_number_of_outputs(self, number_of_outputs):
        for i in range(number_of_outputs):
            output_neuron = OutputNeuron()
            self.add_output_neuron(output_neuron)

    def add_random_connection(self):
        self.brain.add_random_connection()

    def reset_forward_order(self):
        self.brain.forward_order_list = None

    def set_parent(self, parent: 'Genome'):
        self.parent = parent

    def get_reproduction_count(self):
        return self.reproduction_count

    def set_reproduction_count(self, reproduction_count):
        self.reproduction_count = reproduction_count

    # smaller is better
    def get_species_compatibility(self, genome):

        neuron_difference = 0
        connection_difference = 0
        weight_difference = np.abs(self.get_average_weight() - genome.get_average_weight())

        aligned_neurons = Genome.align_gene_ids(self.get_brain().get_all_neurons(),
                                                genome.get_brain().get_all_neurons())
        aligned_connections = Genome.align_gene_ids(self.get_brain().get_all_connections(),
                                                    genome.get_brain().get_all_connections())

        for neuron1, neuron2 in aligned_neurons:
            if neuron1 is None or neuron2 is None:
                neuron_difference += 1

        for connection1, connection2 in aligned_connections:
            if connection1 is None or connection2 is None:
                connection_difference += 1

        return YaneConfig.get_species_compatibility_neuron_factor(yane_config) * neuron_difference + \
            YaneConfig.get_species_compatibility_connection_factor(yane_config) * connection_difference + \
            YaneConfig.get_species_compatibility_weight_factor(yane_config) * weight_difference

    def get_average_weight(self):

        sum_weight = 0

        if len(self.brain.get_all_connections()) == 0:
            return 0

        for connection in self.brain.get_all_connections():
            sum_weight += connection.get_weight()

        return sum_weight / len(self.brain.get_all_connections())

    def clear_hidden_output_neurons(self):
        for neuron in self.brain.get_hidden_neurons() + self.brain.get_output_neurons():
            neuron.set_value(0)

    def get_parent(self):
        return self.parent

    def set_bad_reproduction_count(self, value):
        self.bad_reproduction_count = value

    def get_bad_reproduction_count(self):
        return self.bad_reproduction_count
