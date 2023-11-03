from src.neural_network import YaneConfig
from src.neural_network.InputNeuron import InputNeuron
from src.neural_network.Neuron import Neuron

yane_config = YaneConfig.load_json_config()


class NeuralNetwork:
    def __init__(self):
        self.input_neurons = []
        self.hidden_neurons = []
        self.output_neurons = []

    def add_input_neuron(self, neuron):
        self.input_neurons.append(neuron)

    def add_hidden_neuron(self, neuron):
        self.hidden_neurons.append(neuron)

    def add_output_neuron(self, neuron):
        self.output_neurons.append(neuron)

    def get_input_neurons(self):
        return self.input_neurons

    def get_hidden_neurons(self):
        return self.hidden_neurons

    def get_output_neurons(self):
        return self.output_neurons

    def get_neuron_by_id(self, neuron_id):
        for neuron in self.input_neurons:
            if neuron.get_id() == neuron_id:
                return neuron

        for neuron in self.hidden_neurons:
            if neuron.get_id() == neuron_id:
                return neuron

        for neuron in self.output_neurons:
            if neuron.get_id() == neuron_id:
                return neuron
        return None

    def remove_neuron(self, neuron):
        if neuron in self.input_neurons:
            self.input_neurons.remove(neuron)
        elif neuron in self.hidden_neurons:
            self.hidden_neurons.remove(neuron)
        elif neuron in self.output_neurons:
            self.output_neurons.remove(neuron)

    def set_input_data(self, data):
        while len(data) > len(self.input_neurons):
            new_neuron = InputNeuron()
            self.add_input_neuron(new_neuron)

        for i, v in enumerate(data):
            self.input_neurons[i].set_value(v)

    def forward_propagation(self, data):
        self.clear_values()
        self.set_input_data(data)

        forward_order_list = self.get_forward_order_list()

        for neuron in forward_order_list:
            neuron.fire()

    def clear_values(self):
        if YaneConfig.get_clear_on_new_input(yane_config):
            for neuron in self.input_neurons:
                neuron.set_value(0.0)

            for neuron in self.hidden_neurons:
                neuron.set_value(0.0)

            for neuron in self.output_neurons:
                neuron.set_value(0.0)

    def get_forward_order_list(self):
        forward_order_list = [neuron for neuron in self.input_neurons]

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
        self.custom_evaluate()

    # You need to override this method like this:
    # def custom_evaluate(self):
    #   Your code here to evaluate the neural network or in simple words, to calculate the fitness of the genome
    #   return fitness
    # NeuralNetwork.custom_evaluate = custom_evaluate
    def custom_evaluate(self):
        raise Exception("You need to override the method custom_evaluate(self) in the class NeuralNetwork")

    def get_net_cost(self):
        net_cost = 0.0

        neuron: Neuron

        for neuron in self.input_neurons:
            net_cost += len(neuron.get_next_connections())

        for neuron in self.hidden_neurons:
            net_cost += len(neuron.get_next_connections())

        for neuron in self.output_neurons:
            net_cost += len(neuron.get_next_connections())

        return net_cost
