import bisect
import random
from copy import deepcopy

import networkx as nx
import numpy as np
from line_profiler_pycharm import profile
from matplotlib import pyplot as plt

from src.neural_network.connection.Connection import Connection
from src.neural_network.exceptions.InvalidNode import InvalidNode
from src.neural_network.exceptions.InvalidNodeTypeException import InvalidNodeTypeException
from src.neural_network.node.Node import Node
from src.neural_network.node.NodeTypes import NodeTypes
from src.neural_network.util import YaneConfig

yane_config = YaneConfig.load_json_config()


class Genome:
    def __init__(self, node_genes=None):
        self.bad_reproduction_count = 0
        self.parent: Genome | None = None
        self.fitness = None
        self.net_cost = None
        self.reproduction_count = 0
        self.last_weight_shift_connection = None

        self.relevant_input_nodes = None
        self.next_trigger_nodes = []

        self.input_nodes = []
        self.hidden_nodes = []
        self.output_nodes = []

        self.forward_propagation_finished = False

        self.fired_nodes = set()

        # Mutation probability
        self.mutation_rates = {
            'activation_function_probability': random.random(),  # probability of mutating activation function
            'add_connection_probability': random.random(),  # probability of adding connection
            'remove_connection_probability': random.random(),  # probability of removing connection
            'add_node_probability': random.random(),  # probability of adding node
            'remove_node_probability': random.random(),  # probability of removing node
            'shift_probability': random.random(),  # probability of shifting weight
            'weight_probability': random.random(),  # probability of mutating weight
            'mutation_probability': 0.8,  # probability of mutating a mutation
        }

        # Mutations numbers
        self.mutation_num = {
            "num_structural_mutations_node": 1,  # number of structural mutations
            "num_structural_mutations_connection": 1,  # number of structural mutations
        }

        # TODO: Put connection and node related mutation rates in their respective classes

        if node_genes is not None:
            for node in node_genes:
                self.add_node(node)

    def get_all_connections(self) -> list[Connection]:
        connections = []

        for node in self.get_all_nodes():
            connections.extend(node.get_next_connections().get_all_connections())

        return connections

    def get_last_weight_shift_connection(self) -> Connection:
        return self.last_weight_shift_connection

    def get_all_nodes(self) -> list[Node]:
        return [node for node in self.input_nodes + self.hidden_nodes + self.output_nodes]

    def add_input_node(self, node: Node):
        if node.type is not NodeTypes.INPUT:
            raise InvalidNodeTypeException(
                "Invalid node type. Can only add InputNode")

        bisect.insort(self.input_nodes, node, key=lambda x: x.get_id())

    def add_hidden_node(self, node: Node):
        if node.type is not NodeTypes.HIDDEN:
            raise InvalidNodeTypeException(
                "Invalid node type. Can only add HiddenNode")

        bisect.insort(self.hidden_nodes, node, key=lambda x: x.get_id())

    def add_output_node(self, node: Node):
        if node.type is not NodeTypes.OUTPUT:
            raise InvalidNodeTypeException(
                "Invalid node type. Can only add OutputNode")

        bisect.insort(self.output_nodes, node, key=lambda x: x.get_id())

    def add_node(self, node: Node):
        if self.get_all_nodes().__contains__(node):
            raise InvalidNode("node already exists in the neural network")

        if node.type is NodeTypes.INPUT:
            self.add_input_node(node)
        elif node.type is NodeTypes.HIDDEN:
            self.add_hidden_node(node)
        elif node.type is NodeTypes.OUTPUT:
            self.add_output_node(node)
        else:
            raise InvalidNodeTypeException(
                "Invalid node type. Can only add InputNode, HiddenNode or OutputNode")

    def get_input_nodes(self) -> list[Node]:
        return self.input_nodes

    def get_hidden_nodes(self) -> list[Node]:
        return self.hidden_nodes

    def get_output_nodes(self) -> list[Node]:
        return self.output_nodes

    def get_node_by_id(self, node_id) -> Node | None:
        node: Node

        for node in self.get_all_nodes():
            if node.get_id() == node_id:
                return node

        return None

    def get_random_node(self):
        nodes = self.get_all_nodes()

        if len(nodes) > 0:
            return random.choice(nodes)
        else:
            return None

    def get_input_data(self):
        input_data = []

        for node in self.input_nodes:
            input_data.append(node.get_value())

        return input_data

    def clear_output_nodes(self):
        for node in self.output_nodes:
            node.set_value(0.0)

    def get_relevant_input_nodes(self):
        if self.relevant_input_nodes is not None:
            return self.relevant_input_nodes

        self.relevant_input_nodes = []
        nodes = self.get_output_nodes()

        for node in self.get_output_nodes():
            for previous_node in node.get_previous_connections().get_all_target_nodes():
                if previous_node not in nodes:
                    nodes.append(previous_node)

        self.relevant_input_nodes = [node for node in nodes if node.type is NodeTypes.INPUT]

        return self.relevant_input_nodes

    def add_random_node(self):
        if len(self.get_all_connections()) <= 0:
            return None

        connection: Connection = random.choice(self.get_all_connections())
        source_node: Node = connection.get_source_node()
        target_node: Node = connection.get_target_node()

        new_node = Node(NodeTypes.HIDDEN)
        new_connection1 = Connection(source_node=source_node, target_node=new_node, weight=1)
        new_connection2 = Connection(source_node=new_node, target_node=target_node, weight=connection.get_weight())

        self.add_node(new_node)

        # A ---> C
        # A ---> B ---> C

        self.remove_connection(connection)

        self.add_connection(new_connection1)
        self.add_connection(new_connection2)

        return new_node

    def remove_random_node(self):
        nodes = self.get_hidden_nodes()

        if len(nodes) > 0:
            node = random.choice(nodes)
            self.remove_node(node)

    def remove_node(self, remove_node):
        if remove_node in self.input_nodes:
            self.input_nodes.remove(remove_node)
        elif remove_node in self.hidden_nodes:
            self.hidden_nodes.remove(remove_node)
        elif remove_node in self.output_nodes:
            self.output_nodes.remove(remove_node)

        for connection in remove_node.get_next_connections().get_all_connections():
            self.remove_connection(connection)

        for connection in remove_node.get_previous_connections().get_all_connections():
            self.remove_connection(connection)

    def set_number_of_outputs(self, number_of_outputs):
        for _ in range(number_of_outputs):
            output_node = Node(NodeTypes.OUTPUT)
            self.add_node(output_node)

    def clear_hidden_output_nodes(self):
        for node in self.get_hidden_nodes() + self.get_output_nodes():
            node.set_value(0)

    def get_output_data(self):
        return [node.get_value() for node in self.get_output_nodes()]

    def set_input_data(self, data):
        while len(data) > len(self.input_nodes):
            new_node = Node(NodeTypes.INPUT)
            self.add_input_node(new_node)

        nodes = self.get_relevant_input_nodes()

        for node in nodes:
            if node.get_input_position() >= len(data):
                node.set_value(0.0)
                continue
            node.set_value(data[node.get_input_position()])
            node.set_original_input_data(node.value)

            if node not in self.next_trigger_nodes:
                self.next_trigger_nodes.append(node)

        return nodes

    def plot(self, interactive=False):

        if interactive:
            plt.ion()

        graph = nx.DiGraph()

        edge_colors = []
        node_colors = []
        edge_lengths = []

        for node in self.get_all_nodes():
            graph.add_node(node.get_id(), node_type=str(node.type)[0])
            if node.type == NodeTypes.INPUT:
                node_colors.append('green')
            elif node.type == NodeTypes.HIDDEN:
                node_colors.append('yellow')
            elif node.type == NodeTypes.OUTPUT:
                node_colors.append('red')

        for connection in self.get_all_connections():
            graph.add_edge(connection.get_source_node().get_id(), connection.get_target_node().get_id(),
                           weight=np.round(connection.get_weight(), 2))
            if connection.get_weight() >= 0:
                edge_colors.append('blue')
            else:
                edge_colors.append('red')

            edge_lengths.append(np.abs(connection.get_weight()))

        pos = nx.spring_layout(graph, pos=nx.shell_layout(graph), fixed=None, iterations=50, weight='weight',
                               scale=1.0, k=2, center=None, dim=2, seed=None)
        node_labels = nx.get_node_attributes(graph, 'node_type')
        nx.draw_networkx_labels(graph, pos, labels=node_labels)

        nx.draw_networkx_nodes(graph, pos, node_size=200, node_color=node_colors)
        nx.draw_networkx_edges(graph, pos, arrowsize=20, edge_color=edge_colors)

        labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

        if interactive:
            plt.draw()
            plt.pause(0.001)
            plt.clf()
        else:
            plt.show()

    def mutate(self):
        self.mutate_nodes()
        self.mutate_connections()
        self.mutate_mutation_rates()
        self.mutate_mutation_nums()

    def mutate_nodes(self):
        nodes = self.get_hidden_nodes() + self.get_output_nodes()

        for node in nodes:
            if random.random() < self.mutation_rates['activation_function_probability']:
                node.mutate_activation_function()

        for _ in range(self.mutation_num['num_structural_mutations_node']):
            if random.random() < self.mutation_rates['add_node_probability']:
                self.add_random_node()
            if random.random() < self.mutation_rates['remove_node_probability']:
                self.remove_random_node()

    def mutate_connections(self):
        connections = self.get_all_connections()

        if len(connections) <= 0:
            self.add_random_connection()
            return

        for connection in connections:
            if random.random() < self.mutation_rates['weight_probability']:
                connection.mutate_weight_random()
            if random.random() < self.mutation_rates['shift_probability']:
                self.last_weight_shift_connection = connection.mutate_weight_shift()

        for _ in range(self.mutation_num['num_structural_mutations_connection']):
            if random.random() < self.mutation_rates['add_connection_probability']:
                self.add_random_connection()
            if random.random() < self.mutation_rates['remove_connection_probability']:
                self.remove_random_connection()

    def mutate_mutation_rates(self):
        for rate_name, rate_value in self.mutation_rates.items():
            if random.random() < self.mutation_rates['mutation_probability']:
                change = random.uniform(-0.5, 0.5)
                new_rate = rate_value + change

                self.mutation_rates[rate_name] = min(max(new_rate, 0.01), 1)

    def mutate_mutation_nums(self):
        for num_name, num_value in self.mutation_num.items():
            if random.random() < self.mutation_rates['mutation_probability']:
                change = random.randint(-1, 1)
                new_num_value = num_value + change

                self.mutation_num[num_name] = max(new_num_value, 1)

    # callback_evaluator is a function that takes a genome as a parameter and returns a fitness value
    # This function is used to evaluate the fitness of a genome
    # You have to implement this function yourself since it is specific to your problem
    @profile
    def evaluate(self, callback_evaluator):
        self.set_net_cost(self.calculate_net_cost())
        fitness_result = callback_evaluator(self)

        self.clear_hidden_output_nodes()

        if self.parent is not None and fitness_result >= self.parent.get_fitness():
            self.parent.set_bad_reproduction_count(0)

        # Child genome is worse than parent genome
        if self.parent is not None and fitness_result < self.parent.get_fitness():
            self.parent.set_bad_reproduction_count(self.parent.get_bad_reproduction_count() + 1)

            parent_connection = self.get_parent().get_last_weight_shift_connection()
            if parent_connection is not None:
                parent_connection.switch_weight_shift_direction()

        self.set_fitness(fitness_result)
        return self.get_fitness()

    def remove_random_connection(self):
        connections = self.get_all_connections()

        if len(connections) > 0:
            connection = random.choice(connections)
            self.remove_connection(connection)

    def remove_connection(self, remove_connection: Connection):
        remove_connection.get_source_node().remove_connection(remove_connection)

    def add_random_connection(self):
        source_node: Node = self.get_random_node()
        target_node: Node = self.get_random_node()

        while source_node == target_node:
            target_node = self.get_random_node()

        connection = Connection(source_node, target_node)
        return self.add_connection(connection)

    def add_connection(self, connection: Connection):
        """
        Adds a connection to the neural network in both directions.
        So source node has a next connection and target node has a previous connection
        :param connection:
        :return:
        """

        if connection.get_source_node() is None or connection.get_target_node() is None:
            return None

        if connection.get_source_node() == connection.get_target_node():
            return None

        if self.has_connection(connection):
            return None

        connection.get_source_node().add_connection(connection)
        return connection

    def has_connection(self, connection):
        return connection.get_target_node() in connection.get_source_node().get_next_connections().target_nodes and \
            connection.get_source_node() in connection.get_target_node().get_previous_connections().target_nodes

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
                if random.random() < 0.5:
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

    def get_fitness(self):
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_net_cost(self):
        return self.net_cost

    def set_net_cost(self, net_cost):
        self.net_cost = net_cost

    def tick(self, data=None):
        """
        Every tick the neural network will fire all triggered nodes
        :param data:
        :return:
        """

        trigger_nodes = self.next_trigger_nodes
        self.next_trigger_nodes = []

        if data is not None:
            self.next_trigger_nodes.extend(self.set_input_data(data))

        fired_nodes_length = len(self.fired_nodes)

        if len(self.next_trigger_nodes) == 0:
            self.forward_propagation_finished = True

        for node in trigger_nodes:
            node.fire()
            self.fired_nodes.add(node)

            if fired_nodes_length == len(self.fired_nodes):
                self.forward_propagation_finished = True

            for connection in node.get_next_connections().get_all_connections():
                if connection.get_target_node() not in self.next_trigger_nodes:
                    self.next_trigger_nodes.append(connection.get_target_node())

        return self.get_output_data()

    def forward_propagation(self, data=None):
        """
        :param data: Input data to be set in the input nodes
        :return: Output data from the output nodes
        """
        self.clear_output_nodes()

        if data is not None:
            self.tick(data)

        while not self.forward_propagation_finished:
            self.tick()

        self.fired_nodes.clear()

        return self.get_output_data()

    def calculate_net_cost(self):
        net_cost = len(self.get_all_connections())
        net_cost += len(self.get_all_nodes())
        return net_cost

    def copy(self) -> 'Genome':
        new_genome = Genome()

        for node in self.get_all_nodes():
            new_genome.add_node(node.copy())

        new_genome.mutation_rates = deepcopy(self.mutation_rates)
        new_genome.mutation_num = deepcopy(self.mutation_num)
        new_genome.parent = self

        return new_genome

    def print(self):
        print("Genome: " + str(self.get_fitness()) + " with net cost: " + str(self.get_net_cost()) + " and " + str(
            len(self.get_all_connections())) + " connections")

        print("Mutation rates:")
        for rate_name, rate_value in self.mutation_rates.items():
            print(rate_name + ": " + str(rate_value))

        print("Mutation nums:")
        for num_name, num_value in self.mutation_num.items():
            print(num_name + ": " + str(num_value))

        print("Reproduction count: " + str(self.get_reproduction_count()))
        print("Bad reproduction count: " + str(self.get_bad_reproduction_count()))

        print()

        print("Neural Network:")
        print("Input nodes:")
        for node in self.input_nodes:
            print(node)
        print("Hidden nodes:")
        for node in self.hidden_nodes:
            print(node)
        print("Output nodes:")
        for node in self.output_nodes:
            print(node)
        print("Connections:")
        for connection in self.get_all_connections():
            print(connection)
        print("End of Neural Network")

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

        aligned_nodes = Genome.align_gene_ids(self.get_all_nodes(), genome.get_all_nodes())
        aligned_connections = Genome.align_gene_ids(self.get_all_connections(), genome.get_all_connections())

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

        if len(self.get_all_connections()) == 0:
            return 0

        for connection in self.get_all_connections():
            sum_weight += connection.get_weight()

        return sum_weight / len(self.get_all_connections())

    def get_parent(self):
        return self.parent

    def set_bad_reproduction_count(self, value):
        self.bad_reproduction_count = value

    def get_bad_reproduction_count(self):
        return self.bad_reproduction_count

    def mutate_weight_shift(self):
        connections = self.get_all_connections()

        if len(connections) <= 0:
            return None

        connection = random.choice(connections)
        connection.mutate_weight_shift()

        self.last_weight_shift_connection = connection

        return connection
