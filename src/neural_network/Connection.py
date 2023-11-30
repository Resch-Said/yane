from src.neural_network import YaneConfig
from src.neural_network.Node import Node

yane_config = YaneConfig.load_json_config()


class Connection:
    ID = 0

    def __init__(self, in_node: Node = None, out_node: Node = None, weight: float = 1.0, enabled: bool = True):
        self.weight_shift_direction = True
        self.weight = weight
        self.in_node = in_node
        self.out_node = out_node
        self.enabled = enabled
        self.id = Connection.ID
        Connection.ID += 1

    def set_weight(self, weight):
        self.weight = weight

    def set_in_node(self, neuron):
        self.in_node = neuron

    def set_out_node(self, node):
        self.out_node = node

    def set_enabled(self, enabled):
        self.enabled = enabled

    def get_weight(self):
        return self.weight

    def get_in_node(self) -> Node:
        return self.in_node

    def get_out_node(self) -> Node:
        return self.out_node

    def is_enabled(self) -> bool:
        return self.enabled

    def __str__(self):
        return "Connection: " + str(self.id) + " from " + str(self.in_node.id) + " to " + str(
            self.out_node.id) + " with weight " + str(self.weight) + " and enabled: " + str(self.enabled)

    def get_id(self) -> int:
        return self.id

    def copy(self) -> 'Connection':
        new_connection = Connection(self.in_node, self.out_node, self.weight, self.enabled)
        new_connection.weight_shift_direction = self.weight_shift_direction
        new_connection.id = self.id

        return new_connection

    def mutate_weight_random(self):
        self.weight = YaneConfig.get_random_mutation_weight(yane_config)

    def mutate_enabled(self):
        self.enabled = not self.enabled

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
