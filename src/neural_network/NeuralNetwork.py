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
        self.last_weight_shift_connection = None
        self.input_neurons = []
        self.hidden_neurons = []
        self.output_neurons = []
        self.forward_order_list = None

    def get_all_neurons(self) -> list[Neuron]:
        return [neuron for neuron in self.input_neurons + self.hidden_neurons + self.output_neurons]

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

    def get_input_neurons(self) -> list[InputNeuron]:
        return self.input_neurons

    def get_hidden_neurons(self) -> list[HiddenNeuron]:
        return self.hidden_neurons

    def get_output_neurons(self) -> list[OutputNeuron]:
        return self.output_neurons

    def get_neuron_by_id(self, neuron_id) -> Neuron | None:
        neuron: Neuron

        for neuron in self.get_all_neurons():
            if neuron.get_id() == neuron_id:
                return neuron

        return None

    def get_all_connections(self) -> list[Connection]:
        connections = []
        for neuron in self.get_all_neurons():
            connections += neuron.get_next_connections()
        return connections

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

    def remove_connection(self, remove_connection):
        if remove_connection in self.get_all_connections():
            remove_connection.get_in_neuron().remove_next_connection(remove_connection)

    def set_input_data(self, data):
        while len(data) > len(self.input_neurons):
            new_neuron = InputNeuron()
            self.add_input_neuron(new_neuron)

        for i, v in enumerate(data):
            self.input_neurons[i].set_value(v)
        for i in range(len(data), len(self.input_neurons)):
            self.input_neurons[i].set_value(0.0)

    def forward_propagation(self, data=None):
        self.clear_output()

        if data is not None:
            self.set_input_data(data)

        for neuron in self.get_forward_order_list():
            neuron.fire()

        return self.get_output_data()

    def clear_values(self):
        for neuron in self.hidden_neurons:
            neuron.set_value(0.0)

        for neuron in self.output_neurons:
            neuron.set_value(0.0)

    def get_forward_order_list(self) -> list[Neuron]:

        if self.forward_order_list is not None:
            return self.forward_order_list

        self.forward_order_list = []

        for neuron in self.get_input_neurons():
            if len(neuron.get_next_connections()) > 0:
                self.forward_order_list.append(neuron)

        neuron: Neuron

        for neuron in self.forward_order_list:
            for connection in neuron.get_next_connections():
                if connection.get_out_neuron() not in self.forward_order_list:
                    self.forward_order_list.append(connection.get_out_neuron())

        return self.forward_order_list

    def get_output_data(self) -> list:
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
        self.mutate_neurons()
        self.mutate_connections()

    def mutate_neurons(self):
        neurons = self.get_hidden_neurons() + self.get_output_neurons()

        neuron: Neuron
        for neuron in neurons:
            if random.random() < YaneConfig.get_mutation_activation_function_probability(yane_config):
                neuron.mutate_activation_function()

        if random.random() < YaneConfig.get_mutation_neuron_probability(yane_config):
            self.add_or_remove_random_neuron()

    def mutate_connections(self):
        connections = self.get_all_connections()

        if len(connections) <= 0:
            self.add_random_connection()

        connection: Connection
        for connection in connections:
            if random.random() < YaneConfig.get_mutation_weight_probability(yane_config):
                connection.mutate_weight_random()
            elif random.random() < YaneConfig.get_mutation_enabled_probability(yane_config):
                connection.mutate_enabled()
            elif random.random() < YaneConfig.get_mutation_shift_probability(yane_config):
                self.last_weight_shift_connection = connection.mutate_weight_shift()
            elif random.random() < YaneConfig.get_mutation_connection_probability(yane_config):
                self.add_or_remove_random_connection()

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
            pass

    def remove_random_connection(self):
        connections = self.get_all_connections()

        if len(connections) > 0:
            connection = random.choice(connections)
            self.remove_connection(connection)

    def get_random_neuron(self):
        neurons = self.get_all_neurons()

        if len(neurons) > 0:
            return random.choice(neurons)
        else:
            return None

    def add_random_neuron(self):

        if len(self.get_all_connections()) <= 0:
            return None

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

    def add_or_remove_random_connection(self):
        if random.random() < 0.5:
            self.add_random_connection()
        else:
            self.remove_random_connection()

    def add_or_remove_random_neuron(self):
        if random.random() < 0.5:
            self.add_random_neuron()
        else:
            self.remove_random_neuron()

    def remove_random_neuron(self):
        neurons = self.get_hidden_neurons()

        if len(neurons) > 0:
            neuron = random.choice(neurons)
            self.remove_neuron(neuron)

    def get_last_weight_shift_connection(self) -> Connection:
        return self.last_weight_shift_connection
