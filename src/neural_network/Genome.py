from copy import deepcopy
from random import random

import numpy as np

from src.neural_network import YaneConfig
from src.neural_network.NeuralNetwork import NeuralNetwork
from src.neural_network.Node import Node
from src.neural_network.NodeTypes import NodeTypes

yane_config = YaneConfig.load_json_config()


class Genome:
    def __init__(self, node_genes=None):
        self.bad_reproduction_count = 0
        self.brain: NeuralNetwork = NeuralNetwork()
        self.parent: Genome | None = None
        self.fitness = None
        self.net_cost = None
        self.reproduction_count = 0

        if node_genes is not None:
            for node in node_genes:
                self.add_node(node)

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
            node1 = genes1[index1]
            node2 = genes2[index2]

            if node1.get_id() == node2.get_id():
                aligned_genes.append((node1, node2))
                index1 += 1
                index2 += 1
            elif node1.get_id() < node2.get_id():
                aligned_genes.append((node1, None))
                index1 += 1
            elif node1.get_id() > node2.get_id():
                aligned_genes.append((None, node2))
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
    def crossover_nodes(cls, genome1, genome2) -> list:
        node_genes1 = genome1.get_brain().get_all_nodes()
        node_genes2 = genome2.get_brain().get_all_nodes()

        return Genome.crossover_genes(node_genes1, node_genes2)

    @classmethod
    def crossover(cls, genome1, genome2) -> 'Genome':
        node_genes = Genome.crossover_nodes(genome1, genome2)
        child_genome = Genome(node_genes)

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

        self.clear_hidden_output_nodes()

        # net_cost = self.get_net_cost()

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

    # Avoid deep copy because of recursion
    def copy(self):
        new_genome = Genome()

        for node in self.get_brain().get_all_nodes():
            new_genome.add_node(node.copy())

        for connection in self.get_brain().get_all_connections():
            new_connection = connection.copy()
            new_connection.set_in_node(new_genome.get_brain().get_node_by_id(connection.get_in_node().get_id()))
            new_connection.set_out_node(new_genome.get_brain().get_node_by_id(connection.get_out_node().get_id()))
            new_genome.add_connection(new_connection)

        return new_genome

    def add_node(self, node: Node):
        self.brain.add_node(node)

    def mutate(self):
        self.brain.mutate()

    def print(self):
        print("Genome: " + str(self.get_fitness()) + " with net cost: " + str(self.get_net_cost()) + " and " + str(
            len(self.get_brain().get_forward_order_list())) + " connected nodes")
        self.brain.print()

    def get_all_nodes(self):
        return self.brain.get_all_nodes()

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
            output_node = Node(NodeTypes.OUTPUT)
            self.add_node(output_node)

    def set_parent(self, parent: 'Genome'):
        self.parent = parent

    def get_reproduction_count(self):
        return self.reproduction_count

    def set_reproduction_count(self, reproduction_count):
        self.reproduction_count = reproduction_count

    # smaller is better
    def get_species_compatibility(self, genome):

        node_difference = 0
        connection_difference = 0
        weight_difference = np.abs(self.get_average_weight() - genome.get_average_weight())

        aligned_nodes = Genome.align_gene_ids(self.get_brain().get_all_nodes(),
                                              genome.get_brain().get_all_nodes())
        aligned_connections = Genome.align_gene_ids(self.get_brain().get_all_connections(),
                                                    genome.get_brain().get_all_connections())

        for node1, node2 in aligned_nodes:
            if node1 is None or node2 is None:
                node_difference += 1

        for connection1, connection2 in aligned_connections:
            if connection1 is None or connection2 is None:
                connection_difference += 1

        return YaneConfig.get_species_compatibility_node_factor(yane_config) * node_difference + \
            YaneConfig.get_species_compatibility_connection_factor(yane_config) * connection_difference + \
            YaneConfig.get_species_compatibility_weight_factor(yane_config) * weight_difference

    def get_average_weight(self):

        sum_weight = 0

        if len(self.brain.get_all_connections()) == 0:
            return 0

        for connection in self.brain.get_all_connections():
            sum_weight += connection.get_weight()

        return sum_weight / len(self.brain.get_all_connections())

    def clear_hidden_output_nodes(self):
        for node in self.brain.get_hidden_nodes() + self.brain.get_output_nodes():
            node.set_value(0)

    def get_parent(self):
        return self.parent

    def set_bad_reproduction_count(self, value):
        self.bad_reproduction_count = value

    def get_bad_reproduction_count(self):
        return self.bad_reproduction_count
