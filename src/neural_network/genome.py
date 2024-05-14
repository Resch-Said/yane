import random
from neural_network.node import Node, NodeType
from neural_network.mutation import Mutation
from neural_network.connection import Connection


class Genome:
    def __init__(self) -> None:
        self.fitness = 0
        self.nodes: list[Node] = []
        self.input_nodes: list[Node] = []
        self.output_nodes: list[Node] = []
        self.connected_nodes_forward: dict[Node, list[Connection]] = {}
        self.connected_nodes_backward: dict[Node, list[Connection]] = {}
        self.triggered_nodes: list[Node] = []

        self.mutation_add_node = Mutation()
        self.mutation_add_connection = Mutation()
        self.mutation_remove_node = Mutation()
        self.mutation_remove_connection = Mutation()

        self.mutation_add_node_type = Mutation()

    def tick(self, input_data) -> list[float]:
        """_summary_

        Args:
            input_data (list[float]): Array of input values.

        Returns:
            list[float]: Array of output values.
        """
        self._insert_input_values(input_data)
        self._fire()

        return [node.value for node in self.output_nodes]

    def mutate(self):
        if self.mutation_add_node.mutate_bool(False):
            self._add_node()

        if self.mutation_remove_node.mutate_bool(False):
            self._remove_node()

        if self.mutation_add_connection.mutate_bool(False):
            self._add_connection()

        if self.mutation_remove_connection.mutate_bool(False):
            self._remove_connection()

        for node in self.nodes:
            node.mutate()

        self.mutation_add_node.mutate_rates()
        self.mutation_add_connection.mutate_rates()
        self.mutation_remove_node.mutate_rates()
        self.mutation_remove_connection.mutate_rates()

    def print(self):
        pass

    def copy(self) -> 'Genome':
        genome = Genome()

        for node in self.nodes:
            node_copy = node.copy()
            genome.nodes.append(node_copy)
            if node.type == NodeType.INPUT:
                genome.input_nodes.append(node_copy)
            elif node.type == NodeType.OUTPUT:
                genome.output_nodes.append(node_copy)

        return genome

    def _add_node(self):
        node_type = self.mutation_add_node_type.mutate_custom(
            NodeType.HIDDEN, NodeType)

        node = Node(node_type=node_type)
        if node.type == NodeType.INPUT:
            self.input_nodes.append(node)
        elif node.type == NodeType.OUTPUT:
            self.output_nodes.append(node)

        self.nodes.append(node)

    # TODO: Not finished yet. It's not properly removing the connections and node
    def _remove_node(self):
        node_to_remove = random.choice(self.nodes)

        # Remove all connections to the node
        for connection in self.connected_nodes_backward[node_to_remove]: 
            node_to_remove.disconnect(connection=connection)

        # Remove all connections from the node
        for connection in self.connected_nodes_forward[node_to_remove]:
            node_to_remove.disconnect(connection=connection)

    def _add_connection(self):
        pass

    def _remove_connection(self):
        pass

    def _fire(self):
        """Fires the input nodes with values and the next nodes that got fired.
        """
        triggered_nodes = []

        triggered_nodes.extend(self._fire_triggered_nodes())

        self.triggered_nodes = triggered_nodes

    def _fire_triggered_nodes(self):
        triggered_nodes = []

        for node in self.triggered_nodes:
            triggered_nodes.extend(node.fire())

        return triggered_nodes

    def _insert_input_values(self, input_data):
        data_length = len(input_data)
        triggered_nodes = []

        for node in self.input_nodes:
            if node.input_index <= data_length and node.input_index >= 0:
                node.value = input_data[node.input_index]
                triggered_nodes.append(node)

        self.triggered_nodes.extend(triggered_nodes)
