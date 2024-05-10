from src.neural_network.node.Node import Node
from src.neural_network.util import YaneConfig

yane_config = YaneConfig.load_json_config()


class Connection:
    global_connection_id = 0

    def __init__(self, source_node: Node = None, target_node: Node = None, weight: float = None, shift_direction=True,
                 connection_id: int = None):
        if weight is None:
            weight = YaneConfig.get_random_mutation_weight(yane_config)

        self.weight_shift_direction = shift_direction
        self.weight = weight
        self.source_node = source_node
        self.target_node = target_node
        self.connection_id = connection_id

        if connection_id is None:
            self.connection_id = Connection.global_connection_id
            Connection.global_connection_id += 1

    def set_weight(self, weight):
        self.weight = weight

    def set_source_node(self, neuron):
        self.source_node = neuron

    def set_target_node(self, node):
        self.target_node = node

    def get_weight(self):
        return self.weight

    def get_source_node(self) -> Node:
        return self.source_node

    def get_target_node(self) -> Node:
        return self.target_node

    def __str__(self):
        return "connection: " + str(self.connection_id) + " from " + str(self.source_node.id) + " to " + str(
            self.target_node.id) + " with weight " + str(self.weight)

    def get_id(self) -> int:
        return self.connection_id

    def copy(self) -> 'Connection':
        new_connection = Connection(self.source_node, self.target_node, self.weight, self.weight_shift_direction,
                                    self.connection_id)

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

    def switch_nodes(self) -> 'Connection':
        self.source_node, self.target_node = self.target_node, self.source_node

        return self
