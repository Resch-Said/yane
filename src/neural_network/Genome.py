from copy import deepcopy
from random import random

from src.neural_network import YaneConfig
from src.neural_network.Connection import Connection
from src.neural_network.HiddenNeuron import HiddenNeuron
from src.neural_network.NeuralNetwork import NeuralNetwork, add_connection
from src.neural_network.Neuron import Neuron
from src.neural_network.OutputNeuron import OutputNeuron

yane_config = YaneConfig.load_json_config()


def crossover_neurons(genome1, genome2) -> list:
    genome1_neuron_finished = False
    genome2_neuron_finished = False
    index_pointer_genome1 = 0
    index_pointer_genome2 = 0
    genome1_neuron_size = len(genome1.get_brain().get_all_neurons())
    genome2_neuron_size = len(genome2.get_brain().get_all_neurons())
    genome1_neurons = genome1.get_brain().get_all_neurons()
    genome2_neurons = genome2.get_brain().get_all_neurons()

    neuron_genes = []

    while True:

        if index_pointer_genome1 >= genome1_neuron_size:
            genome1_neuron_finished = True

        if index_pointer_genome2 >= genome2_neuron_size:
            genome2_neuron_finished = True

        if genome1_neuron_finished and genome2_neuron_finished:
            break

        if genome1_neuron_finished:
            neuron_genes.append(genome2_neurons[index_pointer_genome2].copy())
            index_pointer_genome2 += 1
        elif genome2_neuron_finished:
            neuron_genes.append(genome1_neurons[index_pointer_genome1].copy())
            index_pointer_genome1 += 1
        elif genome1_neurons[index_pointer_genome1].get_id() == genome2_neurons[index_pointer_genome2].get_id():
            if random() < 0.5:
                neuron_genes.append(genome1_neurons[index_pointer_genome1].copy())
            else:
                neuron_genes.append(genome2_neurons[index_pointer_genome2].copy())
            index_pointer_genome1 += 1
            index_pointer_genome2 += 1
        elif genome1_neurons[index_pointer_genome1].get_id() < genome2_neurons[index_pointer_genome2].get_id():
            neuron_genes.append(genome1_neurons[index_pointer_genome1].copy())
            index_pointer_genome1 += 1
        elif genome1_neurons[index_pointer_genome1].get_id() > genome2_neurons[index_pointer_genome2].get_id():
            neuron_genes.append(genome2_neurons[index_pointer_genome2].copy())
            index_pointer_genome2 += 1

    return neuron_genes


def crossover_connections(genome1, genome2) -> list:
    index_pointer_genome1 = 0
    index_pointer_genome2 = 0
    genome1_connection_size = len(genome1.get_brain().get_all_connections())
    genome2_connection_size = len(genome2.get_brain().get_all_connections())

    genome1_connections = genome1.get_brain().get_all_connections()
    genome2_connections = genome2.get_brain().get_all_connections()
    genome1_connection_finished = False
    genome2_connection_finished = False

    connection_genes = []

    while True:
        if index_pointer_genome1 >= genome1_connection_size:
            genome1_connection_finished = True

        if index_pointer_genome2 >= genome2_connection_size:
            genome2_connection_finished = True

        if genome1_connection_finished and genome2_connection_finished:
            break

        if genome1_connection_finished:
            connection_genes.append(genome2_connections[index_pointer_genome2].copy())
            index_pointer_genome2 += 1
        elif genome2_connection_finished:
            connection_genes.append(genome1_connections[index_pointer_genome1].copy())
            index_pointer_genome1 += 1
        elif (genome1_connections[index_pointer_genome1].get_id() ==
              genome2_connections[index_pointer_genome2].get_id()):
            if random() < 0.5:
                connection_genes.append(genome1_connections[index_pointer_genome1].copy())
            else:
                connection_genes.append(genome2_connections[index_pointer_genome2].copy())
            index_pointer_genome1 += 1
            index_pointer_genome2 += 1
        elif (genome1_connections[index_pointer_genome1].get_id() <
              genome2_connections[index_pointer_genome2].get_id()):
            connection_genes.append(genome1_connections[index_pointer_genome1].copy())
            index_pointer_genome1 += 1
        elif (genome1_connections[index_pointer_genome1].get_id() >
              genome2_connections[index_pointer_genome2].get_id()):
            connection_genes.append(genome2_connections[index_pointer_genome2].copy())
            index_pointer_genome2 += 1

    return connection_genes


class Genome:
    def __init__(self, neuron_genes=None, connection_genes=None):
        self.brain = NeuralNetwork()
        self.fitness = 0.0
        self.net_cost = 0.0

        if neuron_genes is not None:
            for neuron in neuron_genes:
                self.add_neuron(neuron)

        # TODO: Make sure this works
        if connection_genes is not None:
            for connection in connection_genes:
                for neuron in neuron_genes:
                    if neuron.get_id() == connection.get_in_neuron().get_id():
                        connection.set_in_neuron(neuron)
                    if neuron.get_id() == connection.get_out_neuron().get_id():
                        connection.set_out_neuron(neuron)
                add_connection(connection)

    @classmethod
    def crossover(cls, genome1, genome2) -> 'Genome':
        neuron_genes = crossover_neurons(genome1, genome2)
        connection_genes = crossover_connections(genome1, genome2)

        child_genome = Genome(neuron_genes, connection_genes)
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

    def add_connection(self, connection: Connection):
        add_connection(connection)

    def remove_all_connections(self):
        self.brain.remove_all_connections()

    def mutate(self):
        self.brain.mutate()
