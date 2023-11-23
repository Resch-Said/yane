import random

from src.neural_network import YaneConfig
from src.neural_network.Connection import Connection
from src.neural_network.HiddenNeuron import HiddenNeuron
from src.neural_network.InputNeuron import InputNeuron
from src.neural_network.Neuron import Neuron
from src.neural_network.OutputNeuron import OutputNeuron
from src.neural_network.exceptions.InvalidConnection import InvalidConnection
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

    def calculate_net_cost(self):
        net_cost = len(self.get_all_connections())
        net_cost += len(self.get_all_neurons())
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

        connection = Connection()
        connection.set_in_neuron(random_neuron_in)
        connection.set_out_neuron(random_neuron_out)
        connection.set_weight(YaneConfig.get_random_mutation_weight(yane_config))

        try:
            self.add_connection(connection)
        except InvalidConnection:
            print("Couldn't add random connection. Probably because it already exists")

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

        new_neuron = HiddenNeuron()
        new_connection = Connection()

        self.add_neuron(new_neuron)

        # A ---> C
        # A ---> B ---> C

        new_connection.set_in_neuron(neuron_in)
        new_connection.set_out_neuron(new_neuron)
        connection.set_in_neuron(new_neuron)
        neuron_in.remove_next_connection(connection)
        new_neuron.add_next_connection(connection)
        new_connection.set_weight(1.0)

        self.add_connection(new_connection)

        return new_neuron

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
