from copy import deepcopy, copy

from src.neural_network.ActivationFunction import ActivationFunction
from src.neural_network.Connection import Connection
from src.neural_network.HiddenNeuron import HiddenNeuron
from src.neural_network.InputNeuron import InputNeuron
from src.neural_network.Neuron import Neuron
from src.neural_network.OutputNeuron import OutputNeuron
from src.neural_network.YaneConfig import *


def get_total_fire_rate(working_neurons):
    total_fire_rate = 0
    for neuron in working_neurons:
        total_fire_rate += neuron.fire_rate_variable
    return total_fire_rate


def change_weight_shift_direction(connection):
    connection.weight_shift_direction = not connection.weight_shift_direction


def mutate_weight(random_connection: Connection):
    random_connection.weight = get_mutation_random_weight(json_config)

    NeuralNetwork.last_modified_connection = random_connection


json_config = load_json_config()


class NeuralNetwork:
    def __init__(self, input_neurons_count=None, hidden_neurons_count=None, output_neurons_count=None):

        self.input_neurons = []
        self.hidden_neurons = []
        self.output_neurons = []
        self.connections = []
        self.fitness = 0

        if input_neurons_count is not None:
            for i in range(input_neurons_count):
                self.add_input_neuron(InputNeuron())

        if hidden_neurons_count is not None:
            for i in range(hidden_neurons_count):
                self.add_hidden_neuron(HiddenNeuron())

        if output_neurons_count is not None:
            for i in range(output_neurons_count):
                self.add_output_neuron(OutputNeuron())

    last_modified_connection: Connection = None

    def get_neuron_index(self, neuron):
        if type(neuron) is InputNeuron:
            if self.input_neurons.__contains__(neuron):
                return str(self.input_neurons.index(neuron))
        elif type(neuron) is HiddenNeuron:
            if self.hidden_neurons.__contains__(neuron):
                return str(self.hidden_neurons.index(neuron))
        elif type(neuron) is OutputNeuron:
            if self.output_neurons.__contains__(neuron):
                return str(self.output_neurons.index(neuron))
        return -1

    def get_connection_between_neurons(self, neuron_from: Neuron, neuron_to: Neuron):
        for connection in self.connections:
            if connection.neuron_from == neuron_from and connection.neuron_to == neuron_to:
                return connection
        return None

    def random_mutate_weight(self):
        if len(self.connections) == 0:
            return

        random_connection = random.choice(self.connections)
        mutate_weight(random_connection)

    def optimize_weights(self, fitness_tolerance=0.01):
        self.get_fitness()
        nn_parent = self
        nn_child = nn_parent.create_child()

        fitness_improved = True

        while fitness_improved:
            fitness_improved = False
            for connection in nn_child.connections:
                result = nn_child.optimize_weight_shift(connection, fitness_tolerance)
                if result:
                    fitness_improved = True
        self.copy(nn_child)
        self.get_fitness()

    # First: Optimize weights of parent
    # Second: Create child
    # Third: mutate child
    # Fourth: Optimize weights of child
    # Fifth: Compare fitness of parent and child
    # TODO: Remove training function and move it to NeuroCluster.
    # There is no point to only train 1 neural network and not having a population of neural networks.
    def train(self, min_fitness=-0.1, max_iterations=1000, fitness_tolerance=0.01):
        nn_parent = self
        nn_parent.optimize_weights(fitness_tolerance)
        current_fitness = nn_parent.get_fitness()

        while current_fitness < min_fitness and max_iterations > 0:
            max_iterations -= 1
            nn_child = deepcopy(nn_parent)
            nn_child.mutate()
            nn_child.optimize_weights(fitness_tolerance)

            new_fitness = nn_child.get_fitness()

            if new_fitness >= current_fitness:
                nn_parent = nn_child
                current_fitness = new_fitness
                print("New fitness: " + str(current_fitness))

        self.copy(nn_parent)

    def get_fitness(self):
        self.forward_propagation()
        self.fitness = self.custom_fitness()
        return self.fitness

    def custom_fitness(self):
        fitness = 0
        for neuron in self.output_neurons:
            fitness -= abs(neuron.value - neuron.expected_value)
        return fitness

    def get_last_modified_connection(self):
        for connection in self.connections:
            if connection.deep_id == NeuralNetwork.last_modified_connection.deep_id:
                return connection

    def clear_hidden_neurons(self):
        for neuron in self.hidden_neurons:
            neuron.value = 0.0

    def clear_output_neurons(self):
        for neuron in self.output_neurons:
            neuron.value = 0.0

    def clear_neurons(self):
        self.reset_input_neurons()
        self.reset_fire_rate()

        if get_clear_on_new_input(json_config):
            self.clear_hidden_neurons()
            self.clear_output_neurons()

    def add_input_neuron(self, neuron: Neuron):
        if type(neuron) is not InputNeuron:
            raise TypeError("Neuron is not of type InputNeuron")

        self.input_neurons.append(neuron)

    def add_hidden_neuron(self, neuron: Neuron):
        if type(neuron) is not HiddenNeuron:
            raise TypeError("Neuron is not of type HiddenNeuron")

        self.hidden_neurons.append(neuron)

    def add_output_neuron(self, neuron: Neuron):
        if type(neuron) is not OutputNeuron:
            raise TypeError("Neuron is not of type OutputNeuron")

        self.output_neurons.append(neuron)

    def remove_neuron(self, neuron: Neuron):
        for connection in self.connections:
            if connection.neuron_from == neuron or connection.neuron_to == neuron:
                self.remove_connection(connection)

        if neuron in self.input_neurons:
            self.input_neurons.remove(neuron)
        elif neuron in self.hidden_neurons:
            self.hidden_neurons.remove(neuron)
        elif neuron in self.output_neurons:
            self.output_neurons.remove(neuron)

    def add_connection(self, neuron_from: Neuron, neuron_to: Neuron, weight=1.0):
        connection = Connection(neuron_from, neuron_to, weight)
        self.connections.append(connection)

    def get_connected_neurons_forward(self, neuron: Neuron):
        connected_neurons = []
        for connection in self.connections:
            if connection.neuron_from == neuron:
                connected_neurons.append(connection.neuron_to)

        return connected_neurons

    def get_connected_neurons_backward(self, neuron: Neuron):
        connected_neurons = []
        for connection in self.connections:
            if connection.neuron_to == neuron:
                connected_neurons.append(connection.neuron_from)
        return connected_neurons

    def remove_connection_between_neurons(self, neuron_from: Neuron, neuron_to: Neuron):
        for connection in self.connections:
            if connection.neuron_from == neuron_from and connection.neuron_to == neuron_to:
                self.remove_connection(connection)

    def get_connection_weight(self, neuron_from, neuron_to):
        for connection in self.connections:
            if connection.neuron_from == neuron_from and connection.neuron_to == neuron_to:
                return connection.weight

    def get_neuron_forward_order(self):
        working_neurons = copy(self.input_neurons)

        for neuron in working_neurons:
            for next_neuron in self.get_connected_neurons_forward(neuron):
                if next_neuron not in working_neurons:
                    working_neurons.append(next_neuron)

        return working_neurons

    def reset_input_neurons(self):
        for neuron in self.input_neurons:
            neuron.value = neuron.value_fixed

    def get_output_values(self):
        output_values = []
        for neuron in self.output_neurons:
            output_values.append(neuron.value)
        return output_values

    def get_expected_output_values(self):
        expected_output_values = []
        for neuron in self.output_neurons:
            expected_output_values.append(neuron.expected_value)
        return expected_output_values

    def set_expected_output_values(self, param):

        while len(param) > len(self.output_neurons):
            new_neuron = OutputNeuron()
            self.add_output_neuron(new_neuron)

        for expected_value, neuron in zip(param, self.output_neurons):
            neuron.expected_value = expected_value

    def copy(self, nn_current):
        self.input_neurons = copy(nn_current.input_neurons)
        self.hidden_neurons = copy(nn_current.hidden_neurons)
        self.output_neurons = copy(nn_current.output_neurons)
        self.connections = copy(nn_current.connections)
        self.fitness = copy(nn_current.fitness)

    def reset_fire_rate(self):
        for neuron in self.input_neurons:
            neuron.fire_rate_variable = neuron.fire_rate_fixed
        for neuron in self.hidden_neurons:
            neuron.fire_rate_variable = neuron.fire_rate_fixed
        for neuron in self.output_neurons:
            neuron.fire_rate_variable = neuron.fire_rate_fixed

    def set_all_activation_functions(self, linear):
        for neuron in self.input_neurons:
            neuron.activation_function = linear
        for neuron in self.hidden_neurons:
            neuron.activation_function = linear
        for neuron in self.output_neurons:
            neuron.activation_function = linear

    def forward_propagation(self):  # One tick cycle
        self.clear_neurons()
        working_neurons = self.get_neuron_forward_order()

        while working_neurons:
            for neuron in working_neurons[:]:
                if neuron.fire_rate_variable > 0:
                    neuron.fire_rate_variable -= 1
                    ActivationFunction.activate(neuron)
                else:
                    working_neurons.remove(neuron)
                for connection in self.get_connections(neuron):
                    connection.neuron_to.value += neuron.value * connection.weight

    def get_connections(self, neuron):
        connections = [connection for connection in self.connections if connection.neuron_from == neuron]
        return connections

    def create_child(self):
        nn_child = deepcopy(self)
        return nn_child

    def mutate(self):
        if random.random() < get_mutation_weight_probability(json_config):
            self.random_mutate_weight()

        if random.random() < get_mutation_connection_probability(json_config):
            self.random_mutate_connection()

        if random.random() < get_mutation_activation_function_probability(json_config):
            self.random_mutate_activation_function()

        if random.random() < get_mutation_fire_rate_probability(json_config):
            self.random_mutate_fire_rate()

        if random.random() < get_mutation_neuron_probability(json_config):
            self.random_mutate_neuron()

    def print(self):
        self.reset_input_neurons()
        self.reset_fire_rate()

        print("Input neurons: " + str(len(self.input_neurons)))
        print("Hidden neurons: " + str(len(self.hidden_neurons)))
        print("Output neurons: " + str(len(self.output_neurons)))
        print("Connections: " + str(len(self.connections)))
        print("")

        for neuron in self.input_neurons:
            print("Input neuron: " + str(neuron.value) + " | activation: " + str(
                neuron.activation_function) + " | fire rate: " + str(neuron.fire_rate_variable))

        for neuron in self.hidden_neurons:
            print("Hidden neuron: " + str(neuron.value) + " | activation: " + str(
                neuron.activation_function) + " | fire rate: " + str(neuron.fire_rate_variable))

        for neuron in self.output_neurons:
            print("Output neuron: " + str(neuron.value) + " | activation: " + str(
                neuron.activation_function) + " | fire rate: " + str(neuron.fire_rate_variable))

        for connection in self.connections:
            print("Connection: " + str(connection.weight) + " | from: " + str(
                type(connection.neuron_from).__name__) + str(
                self.get_neuron_index(connection.neuron_from)) + " | to: " + str(
                type(connection.neuron_to).__name__) + str(self.get_neuron_index(connection.neuron_to)))

    def set_input_neurons(self, param):

        while len(param) > len(self.input_neurons):
            new_neuron = InputNeuron()
            self.add_input_neuron(new_neuron)

        for input_value, neuron in zip(param, self.input_neurons):
            neuron.value_fixed = input_value

    def random_mutate_connection(self):
        if random.random() < 0.5:
            self.create_random_connection()
        else:
            self.remove_random_connection()

    def create_random_connection(self):
        random_neuron_from = self.get_random_neuron()
        random_neuron_to = self.get_random_neuron()
        if not self.connections.__contains__(self.get_connection_between_neurons(random_neuron_from, random_neuron_to)):
            self.add_connection(random_neuron_from, random_neuron_to, 0)

    def get_random_neuron(self) -> Neuron:
        random_neuron = random.choice(self.input_neurons + self.hidden_neurons + self.output_neurons)
        return random_neuron

    def remove_random_connection(self):
        if len(self.connections) > 0:
            random_connection = random.choice(self.connections)
            self.remove_connection(random_connection)

    # TODO: Make get_random_weight_shift less random and more intelligent
    # Example: Start with a big weight shift and then decrease the weight shift
    def optimize_weight_shift(self, connection, fitness_tolerance=0.01):
        old_fitness = self.get_fitness()
        fitness_improved_up = True
        fitness_improved_down = True
        fitness_improved = False
        old_weight = connection.weight

        while fitness_improved_up or fitness_improved_down:
            if connection.weight_shift_direction:
                connection.weight += get_random_weight_shift(json_config)
            else:
                connection.weight -= get_random_weight_shift(json_config)

            new_fitness = self.get_fitness()

            if new_fitness > old_fitness and abs(new_fitness - old_fitness) > fitness_tolerance:
                fitness_improved = True
                old_fitness = new_fitness
                old_weight = connection.weight
                fitness_improved_up = True
                fitness_improved_down = True
            else:
                connection.weight = old_weight
                if connection.weight_shift_direction:
                    fitness_improved_up = False
                else:
                    fitness_improved_down = False
                connection.weight_shift_direction = not connection.weight_shift_direction

        NeuralNetwork.last_modified_connection = connection
        return fitness_improved

    def random_mutate_activation_function(self):
        random_neuron = self.get_random_neuron()
        random_neuron.activation_function = get_random_activation_function(json_config)

    def random_mutate_fire_rate(self):
        random_neuron = self.get_random_neuron()
        random_neuron.fire_rate_fixed = get_random_fire_rate(json_config)
        random_neuron.fire_rate_variable = random_neuron.fire_rate_fixed

    def random_mutate_neuron(self):
        if random.random() < 0.5:
            self.create_random_neuron()
        else:
            self.remove_random_neuron()

    # Neurons are only created between connected neurons
    def create_random_neuron(self):
        random_neuron_from = self.get_random_neuron()
        random_neuron_to = self.get_random_neuron()
        new_neuron = HiddenNeuron()

        connection = self.get_connection_between_neurons(random_neuron_from, random_neuron_to)
        if connection is None:
            return

        self.add_hidden_neuron(new_neuron)

        self.add_connection(random_neuron_from, new_neuron, 1)
        self.add_connection(new_neuron, random_neuron_to, connection.weight)

        self.remove_connection(connection)

    # Does not remove input or output neurons
    def remove_random_neuron(self):
        random_neuron = self.get_random_hidden_neuron()

        if random_neuron is not None:
            self.remove_neuron(random_neuron)

    def get_random_hidden_neuron(self):
        if len(self.hidden_neurons) == 0:
            return None

        random_neuron = random.choice(self.hidden_neurons)
        return random_neuron

    def remove_connection(self, connection):
        self.connections.remove(connection)
