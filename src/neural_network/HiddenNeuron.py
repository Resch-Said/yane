from src.neural_network.Neuron import Neuron


class HiddenNeuron(Neuron):
    def __init__(self):
        super().__init__()

    def fire(self):
        self.activate()

        for connection in self.next_connections:
            next_neuron: Neuron = connection.get_out_neuron()
            next_neuron.set_value(next_neuron.get_value() + self.value * connection.get_weight())

        self.value = 0.0
