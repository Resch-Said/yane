from src.neural_network.Neuron import Neuron


class Connection:
    deep_id = 0

    def __init__(self, neuron_from: Neuron, neuron_to: Neuron, weight=1.0):
        self.neuron_from = neuron_from
        self.neuron_to = neuron_to
        self.weight = weight
        self.weight_shift_direction = True  # True = Up, False = Down
        self.deep_id = Connection.deep_id  # For comparing connections between parent and child
        Connection.deep_id += 1
