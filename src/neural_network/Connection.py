from src.neural_network import YaneConfig
from src.neural_network.Node import Node

yane_config = YaneConfig.load_json_config()


class Connection:
    ID = 0

    def __init__(self, in_node: Node = None, out_node: Node = None, weight: float = None, ID=None):

        if weight is None:
            weight = YaneConfig.get_random_mutation_weight(yane_config)
        self.weight_shift_direction = True
        self.weight = weight
        self.in_node = in_node
        self.out_node = out_node
        self.id = ID

        if ID is None:
            self.id = Connection.ID
            Connection.ID += 1

    def set_weight(self, weight):
        self.weight = weight

    def set_in_node(self, neuron):
        self.in_node = neuron

    def set_out_node(self, node):
        self.out_node = node

    def get_weight(self):
        return self.weight

    def get_in_node(self) -> Node:
        return self.in_node

    def get_out_node(self) -> Node:
        return self.out_node

    def __str__(self):
        return "Connection: " + str(self.id) + " from " + str(self.in_node.id) + " to " + str(
            self.out_node.id) + " with weight " + str(self.weight)

    def get_id(self) -> int:
        return self.id

    def copy(self) -> 'Connection':
        new_connection = Connection(self.in_node, self.out_node, self.weight, self.id)
        new_connection.weight_shift_direction = self.weight_shift_direction

        return new_connection

    def mutate_weight_random(self):
        self.weight = YaneConfig.get_random_mutation_weight(yane_config)

    def mutate_weight_shift(self):

        weight_shift = YaneConfig.get_random_weight_shift(yane_config)

        if self.get_weight_shift_direction():
            self.weight += weight_shift
        else:
            self.weight -= weight_shift

        if self.weight < YaneConfig.get_mutation_weight_min(yane_config):
            self.weight = YaneConfig.get_mutation_weight_min(yane_config)
        elif self.weight > YaneConfig.get_mutation_weight_max(yane_config):
            self.weight = YaneConfig.get_mutation_weight_max(yane_config)

        return self

    def get_weight_shift_direction(self):
        return self.weight_shift_direction

    def switch_weight_shift_direction(self):
        self.weight_shift_direction = not self.weight_shift_direction
