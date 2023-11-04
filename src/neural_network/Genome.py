from copy import deepcopy
from random import random

from src.neural_network import YaneConfig
from src.neural_network.HiddenNeuron import HiddenNeuron
from src.neural_network.NeuralNetwork import NeuralNetwork, add_connection
from src.neural_network.Neuron import Neuron
from src.neural_network.OutputNeuron import OutputNeuron

yane_config = YaneConfig.load_json_config()


class Genome:
    def __init__(self, neuron_genes=None, connection_genes=None):
        self.brain = NeuralNetwork()
        self.fitness = 0.0
        self.net_cost = 0.0

        if neuron_genes is not None:
            for neuron in neuron_genes:
                self.add_neuron(neuron)

        # TODO: Make sure this works
        self.combine_neuron_connection_genes(connection_genes, neuron_genes)

    @classmethod
    def crossover(cls, genome1, genome2) -> 'Genome':
        neuron_genes = cls.crossover_neurons(genome1, genome2)
        connection_genes = cls.crossover_connections(genome1, genome2)

        child_genome = Genome(neuron_genes, connection_genes)
        return child_genome

    @classmethod
    def crossover_neurons(cls, genome1, genome2) -> list:

        neuron_genes1 = genome1.get_brain().get_all_neurons()
        neuron_genes2 = genome2.get_brain().get_all_neurons()

        return cls.crossover_genes(neuron_genes1, neuron_genes2)

    @classmethod
    def crossover_genes(cls, gene1, gene2) -> list:
        iter_gene1 = iter(gene1)
        iter_gene2 = iter(gene2)
        gene1_gene = next(iter_gene1, None)
        gene2_gene = next(iter_gene2, None)

        new_genes = []

        while gene1_gene is not None and gene2_gene is not None:
            if gene1_gene.get_id() == gene2_gene.get_id():
                if random() < 0.5:
                    new_genes.append(gene1_gene.copy())
                else:
                    new_genes.append(gene2_gene.copy())

                gene1_gene = next(iter_gene1, None)
                gene2_gene = next(iter_gene2, None)
            elif gene1_gene.get_id() < gene2_gene.get_id():
                new_genes.append(gene1_gene.copy())
                gene1_gene = next(iter_gene1, None)
            elif gene1_gene.get_id() > gene2_gene.get_id():
                new_genes.append(gene2_gene.copy())
                gene2_gene = next(iter_gene2, None)

        return new_genes

    @classmethod
    def crossover_connections(cls, genome1, genome2) -> list:
        return cls.crossover_genes(genome1, genome2)

    @classmethod
    def combine_neuron_connection_genes(cls, connection_genes, neuron_genes):
        if connection_genes is not None and neuron_genes is not None:
            for connection in connection_genes:
                for neuron in neuron_genes:
                    if neuron.get_id() == connection.get_in_neuron().get_id():
                        connection.set_in_neuron(neuron)
                    if neuron.get_id() == connection.get_out_neuron().get_id():
                        connection.set_out_neuron(neuron)
                add_connection(connection)

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

    def evaluate(self):
        self.set_net_cost(self.get_brain().calculate_net_cost())
        self.set_fitness(
            self.get_brain().evaluate() - self.get_net_cost() * YaneConfig.get_net_cost_factor(yane_config))
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
