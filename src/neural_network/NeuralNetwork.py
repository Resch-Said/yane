import random

from src.neural_network import YaneConfig
from src.neural_network.Connection import Connection
from src.neural_network.HiddenNeuron import HiddenNeuron
from src.neural_network.InputNeuron import InputNeuron
from src.neural_network.Neuron import Neuron
from src.neural_network.OutputNeuron import OutputNeuron
from src.neural_network.exceptions.InvalidNeuron import InvalidNeuron
from src.neural_network.exceptions.InvalidNeuronTypeException import InvalidNeuronTypeException

yane_config = YaneConfig.load_json_config()


class NeuralNetwork:
    def __init__(self):
        self.input_neurons = []
        self.hidden_neurons = []
        self.output_neurons = []

        bias_neuron = InputNeuron()
        bias_neuron.set_value(1.0)
        self.add_input_neuron(bias_neuron)

    def get_all_neurons(self) -> list:
        return sorted([neuron for neuron in self.input_neurons + self.hidden_neurons + self.output_neurons],
                      key=lambda x: x.get_id())

    def add_connection(self, connection):
        if self.get_all_neurons().__contains__(connection.get_in_neuron()) is False:
            raise InvalidNeuron("Neuron in is not in the neural network")

        if self.get_all_neurons().__contains__(connection.get_out_neuron()) is False:
            raise InvalidNeuron("Neuron out is not in the neural network")

        connection.get_in_neuron().add_next_connection(connection)

    def add_input_neuron(self, neuron: InputNeuron):
        if not isinstance(neuron, InputNeuron):
            raise InvalidNeuronTypeException(
                "Invalid neuron type. Can only add InputNeuron")

        self.input_neurons.append(neuron)

    def add_hidden_neuron(self, neuron: HiddenNeuron):
        if not isinstance(neuron, HiddenNeuron):
            raise InvalidNeuronTypeException(
                "Invalid neuron type. Can only add HiddenNeuron")

        self.hidden_neurons.append(neuron)

    def add_output_neuron(self, neuron: OutputNeuron):
        if not isinstance(neuron, OutputNeuron):
            raise InvalidNeuronTypeException(
                "Invalid neuron type. Can only add OutputNeuron")

        self.output_neurons.append(neuron)

    def add_neuron(self, neuron: Neuron):

        if self.get_all_neurons().__contains__(neuron):
            raise InvalidNeuron("Neuron already exists in the neural network")

        if isinstance(neuron, InputNeuron):
            self.add_input_neuron(neuron)
        elif isinstance(neuron, HiddenNeuron):
            self.add_hidden_neuron(neuron)
        elif isinstance(neuron, OutputNeuron):
            self.add_output_neuron(neuron)
        else:
            raise InvalidNeuronTypeException(
                "Invalid neuron type. Can only add InputNeuron, HiddenNeuron or OutputNeuron")

    def get_input_neurons(self):
        return self.input_neurons

    def get_hidden_neurons(self):
        return self.hidden_neurons

    def get_output_neurons(self):
        return self.output_neurons

    def get_neuron_by_id(self, neuron_id):
        for neuron in self.get_all_neurons():
            if neuron.get_id() == neuron_id:
                return neuron

        return None

    def get_all_connections(self):
        connections = []
        for neuron in self.get_all_neurons():
            connections += neuron.get_next_connections()
        return sorted(list(set(connections)), key=lambda x: x.get_id())

    def remove_neuron(self, remove_neuron):
        if remove_neuron in self.input_neurons:
            self.input_neurons.remove(remove_neuron)
        elif remove_neuron in self.hidden_neurons:
            self.hidden_neurons.remove(remove_neuron)
        elif remove_neuron in self.output_neurons:
            self.output_neurons.remove(remove_neuron)

        for neuron in self.get_all_neurons():
            for con in neuron.get_next_connections():
                if con.get_out_neuron() == remove_neuron:
                    neuron.remove_next_connection(con)

    def set_input_data(self, data):
        while len(data) > len(self.input_neurons) - 1:
            new_neuron = InputNeuron()
            self.add_input_neuron(new_neuron)

        for i, v in enumerate(data):
            self.input_neurons[i + 1].set_value(v)
        for i in range(len(data), len(self.input_neurons) - 1):
            self.input_neurons[i + 1].set_value(0.0)

    def forward_propagation(self, data=None):
        self.clear_output()

        if data is not None:
            self.set_input_data(data)

        forward_order_list = self.get_forward_order_list()

        for neuron in forward_order_list:
            neuron.fire()

        return self.get_output_data()

    def clear_values(self):
        for neuron in self.hidden_neurons:
            neuron.set_value(0.0)

        for neuron in self.output_neurons:
            neuron.set_value(0.0)

    def get_forward_order_list(self) -> list:

        forward_order_list = []

        for neuron in self.get_input_neurons():
            if len(neuron.get_next_connections()) > 0:
                forward_order_list.append(neuron)

        neuron: Neuron

        for neuron in forward_order_list:
            for connection in neuron.get_next_connections():
                if connection.get_out_neuron() not in forward_order_list:
                    forward_order_list.append(connection.get_out_neuron())

        return forward_order_list

    def get_output_data(self):
        output_data = []

        for neuron in self.output_neurons:
            output_data.append(neuron.get_value())

        return output_data

    def evaluate(self):
        return self.custom_evaluation()

    # You need to override this method like this:
    # def custom_evaluate(self):
    #   Your code here to evaluate the neural network or in simple words, to calculate the fitness of the genome
    #   remember to also call the method forward_propagation(self, data) to set the input data and calculate the output
    #   return fitness
    # NeuralNetwork.custom_evaluate = custom_evaluate
    def custom_evaluation(self):
        raise Exception("You need to override the method custom_evaluate(self) in the class NeuralNetwork")

    def calculate_net_cost(self):
        net_cost = 0.0
        neuron: Neuron

        for neuron in self.input_neurons:
            net_cost += len(neuron.get_next_connections())

        for neuron in self.hidden_neurons:
            net_cost += len(neuron.get_next_connections())

        for neuron in self.output_neurons:
            net_cost += len(neuron.get_next_connections())

        return net_cost

    def remove_all_connections(self):
        for neuron in self.get_all_neurons():
            neuron.next_connections = []

    def mutate(self):
        self.mutate_neuron_genes()
        self.mutate_connection_genes()

    def mutate_neuron_genes(self):
        neuron_genes = self.get_all_neurons()

        neuron: Neuron
        for neuron in neuron_genes:
            if random.random() < YaneConfig.get_mutation_activation_function_probability(yane_config):
                neuron.mutate_activation_function()
            elif random.random() < YaneConfig.get_mutation_neuron_probability(yane_config):
                self.add_random_neuron()

    def mutate_connection_genes(self):
        connection_genes = self.get_all_connections()

        if len(connection_genes) <= 0:
            self.add_random_connection()

        connection: Connection
        for connection in connection_genes:
            if random.random() < YaneConfig.get_mutation_weight_probability(yane_config):
                connection.mutate_weight()
            elif random.random() < YaneConfig.get_mutation_enabled_probability(yane_config):
                connection.mutate_enabled()
            elif random.random() < YaneConfig.get_mutation_shift_probability(yane_config):
                connection.mutate_weight_shift()
            elif random.random() < YaneConfig.get_mutation_connection_probability(yane_config):
                self.add_random_connection()

    def add_random_connection(self):
        random_neuron_in: Neuron = self.get_random_neuron()
        random_neuron_out: Neuron = self.get_random_neuron()

        if random_neuron_in.get_next_connections().__contains__(random_neuron_out):
            return

        if random_neuron_in is None or random_neuron_out is None:
            raise Exception("Neuron in or neuron out is None")

        connection = Connection()
        connection.set_in_neuron(random_neuron_in)
        connection.set_out_neuron(random_neuron_out)
        connection.set_weight(YaneConfig.get_random_mutation_weight(yane_config))
        NeuralNetwork.add_connection(connection)

    def get_random_neuron(self):
        neurons = self.get_all_neurons()

        if len(neurons) > 0:
            return random.choice(neurons)
        else:
            return None

    def add_random_neuron(self):
        if len(self.get_all_connections()) == 0:
            return

        connection = random.choice(self.get_all_connections())
        neuron_in: Neuron = connection.get_in_neuron()
        neuron_out: Neuron = connection.get_out_neuron()

        if neuron_in is None or neuron_out is None:
            raise Exception("Neuron in or neuron out is None")

        new_neuron = HiddenNeuron()
        new_connection = Connection()

        # A ---> B
        # A ---> C
        # A ---> B ---> C

        new_connection.set_in_neuron(neuron_in)
        new_connection.set_out_neuron(new_neuron)
        connection.set_in_neuron(new_neuron)

        new_connection.set_weight(1.0)

        NeuralNetwork.add_connection(new_connection)
        neuron_in.get_next_connections().remove(connection)

    def print(self):
        print("Neural Network:")
        print("Input Neurons:")
        for neuron in self.input_neurons:
            print(neuron)
        print("Hidden Neurons:")
        for neuron in self.hidden_neurons:
            print(neuron)
        print("Output Neurons:")
        for neuron in self.output_neurons:
            print(neuron)
        print("Connections:")
        for connection in self.get_all_connections():
            print(connection)
        print("End of Neural Network")

    def get_input_data(self):
        input_data = []

        for neuron in self.input_neurons:
            input_data.append(neuron.get_value())

        return input_data

    def clear_output(self):
        for neuron in self.output_neurons:
            neuron.set_value(0.0)
