import bisect
import random

from src.neural_network import YaneConfig
from src.neural_network.Connection import Connection
from src.neural_network.Node import Node
from src.neural_network.NodeTypes import NodeTypes
from src.neural_network.exceptions.InvalidConnection import InvalidConnection
from src.neural_network.exceptions.InvalidNode import InvalidNode
from src.neural_network.exceptions.InvalidNodeTypeException import InvalidNodeTypeException

yane_config = YaneConfig.load_json_config()


class NeuralNetwork:
    def __init__(self):
        self.last_weight_shift_connection = None
        self.input_nodes = []
        self.hidden_nodes = []
        self.output_nodes = []
        self.forward_order_list = None

    def get_all_nodes(self) -> list[Node]:
        return [node for node in self.input_nodes + self.hidden_nodes + self.output_nodes]

    def add_connection(self, connection):
        if self.get_all_nodes().__contains__(connection.get_in_node()) is False:
            raise InvalidNode("node in is not in the neural network")

        if self.get_all_nodes().__contains__(connection.get_out_node()) is False:
            raise InvalidNode("node out is not in the neural network")

        connection.get_in_node().add_next_connection(connection)

    def add_input_node(self, node: Node):
        if node.type is not NodeTypes.INPUT:
            raise InvalidNodeTypeException(
                "Invalid node type. Can only add InputNode")

        bisect.insort(self.input_nodes, node, key=lambda x: x.get_id())

    def add_hidden_node(self, node: Node):
        if node.type is not NodeTypes.HIDDEN:
            raise InvalidNodeTypeException(
                "Invalid node type. Can only add HiddenNode")

        bisect.insort(self.hidden_nodes, node, key=lambda x: x.get_id())

    def add_output_node(self, node: Node):
        if node.type is not NodeTypes.OUTPUT:
            raise InvalidNodeTypeException(
                "Invalid node type. Can only add OutputNode")

        bisect.insort(self.output_nodes, node, key=lambda x: x.get_id())

    def add_node(self, node: Node):

        if self.get_all_nodes().__contains__(node):
            raise InvalidNode("node already exists in the neural network")

        if node.type is NodeTypes.INPUT:
            self.add_input_node(node)
        elif node.type is NodeTypes.HIDDEN:
            self.add_hidden_node(node)
        elif node.type is NodeTypes.OUTPUT:
            self.add_output_node(node)
        else:
            raise InvalidNodeTypeException(
                "Invalid node type. Can only add InputNode, HiddenNode or OutputNode")

    def get_input_nodes(self) -> list[Node]:
        return self.input_nodes

    def get_hidden_nodes(self) -> list[Node]:
        return self.hidden_nodes

    def get_output_nodes(self) -> list[Node]:
        return self.output_nodes

    def get_node_by_id(self, node_id) -> Node | None:
        node: Node

        for node in self.get_all_nodes():
            if node.get_id() == node_id:
                return node

        return None

    def get_all_connections(self) -> list[Connection]:
        connections = []
        for node in self.get_all_nodes():
            connections += node.get_next_connections()
        return connections

    def remove_node(self, remove_node):
        if remove_node in self.input_nodes:
            self.input_nodes.remove(remove_node)
        elif remove_node in self.hidden_nodes:
            self.hidden_nodes.remove(remove_node)
        elif remove_node in self.output_nodes:
            self.output_nodes.remove(remove_node)

        for node in self.get_all_nodes():
            for con in node.get_next_connections():
                if con.get_out_node() == remove_node:
                    node.remove_next_connection(con)

    def remove_connection(self, remove_connection):
        if remove_connection in self.get_all_connections():
            remove_connection.get_in_node().remove_next_connection(remove_connection)

    def set_input_data(self, data):
        while len(data) > len(self.input_nodes):
            new_node = Node(NodeTypes.INPUT)
            self.add_input_node(new_node)

        for i, v in enumerate(data):
            self.input_nodes[i].set_value(v)
        for i in range(len(data), len(self.input_nodes)):
            self.input_nodes[i].set_value(0.0)

    def forward_propagation(self, data=None):
        self.clear_output()

        if data is not None:
            self.set_input_data(data)

        for node in self.get_forward_order_list():
            node.fire()

        return self.get_output_data()

    def clear_values(self):
        for node in self.hidden_nodes:
            node.set_value(0.0)

        for node in self.output_nodes:
            node.set_value(0.0)

    def get_forward_order_list(self) -> list[Node]:

        if self.forward_order_list is not None:
            return self.forward_order_list

        self.forward_order_list = []

        for node in self.get_input_nodes():
            if len(node.get_next_connections()) > 0:
                self.forward_order_list.append(node)

        node: Node

        for node in self.forward_order_list:
            for connection in node.get_next_connections():
                if connection.get_out_node() not in self.forward_order_list:
                    self.forward_order_list.append(connection.get_out_node())

        return self.forward_order_list

    def get_output_data(self) -> list:
        output_data = []

        for node in self.output_nodes:
            output_data.append(node.get_value())

        return output_data

    def calculate_net_cost(self):
        net_cost = len(self.get_all_connections())
        net_cost += len(self.get_all_nodes())
        return net_cost

    def remove_all_connections(self):
        for node in self.get_all_nodes():
            node.next_connections = []

    def mutate(self):
        self.mutate_nodes()
        self.mutate_connections()

    def mutate_nodes(self):
        nodes = self.get_hidden_nodes() + self.get_output_nodes()
        random_node = random.choice(nodes)

        if random.random() < YaneConfig.get_mutation_activation_function_probability(yane_config):
            random_node.mutate_activation_function()

        if random.random() < YaneConfig.get_mutation_node_probability(yane_config):
            self.add_or_remove_random_node()

    def mutate_connections(self):
        connections = self.get_all_connections()

        if len(connections) <= 0:
            self.add_random_connection()
            return

        random_connection = random.choice(connections)

        if random.random() < YaneConfig.get_mutation_weight_probability(yane_config):
            random_connection.mutate_weight_random()
        if random.random() < YaneConfig.get_mutation_enabled_probability(yane_config):
            random_connection.mutate_enabled()
        if random.random() < YaneConfig.get_mutation_shift_probability(yane_config):
            self.last_weight_shift_connection = random_connection.mutate_weight_shift()
        if random.random() < YaneConfig.get_mutation_connection_probability(yane_config):
            self.add_or_remove_random_connection()

    def add_random_connection(self):
        random_node_in: Node = self.get_random_node()
        random_node_out: Node = self.get_random_node()

        connection = Connection()
        connection.set_in_node(random_node_in)
        connection.set_out_node(random_node_out)
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

    def get_random_node(self):
        nodes = self.get_all_nodes()

        if len(nodes) > 0:
            return random.choice(nodes)
        else:
            return None

    def add_random_node(self):

        if len(self.get_all_connections()) <= 0:
            return None

        connection = random.choice(self.get_all_connections())
        node_in: Node = connection.get_in_node()

        new_node = Node(NodeTypes.HIDDEN)
        new_connection = Connection()

        self.add_node(new_node)

        # A ---> C
        # A ---> B ---> C

        new_connection.set_in_node(node_in)
        new_connection.set_out_node(new_node)
        connection.set_in_node(new_node)
        node_in.remove_next_connection(connection)
        new_node.add_next_connection(connection)
        new_connection.set_weight(1.0)

        self.add_connection(new_connection)

        return new_node

    def print(self):
        print("Neural Network:")
        print("Input nodes:")
        for node in self.input_nodes:
            print(node)
        print("Hidden nodes:")
        for node in self.hidden_nodes:
            print(node)
        print("Output nodes:")
        for node in self.output_nodes:
            print(node)
        print("Connections:")
        for connection in self.get_all_connections():
            print(connection)
        print("End of Neural Network")

    def get_input_data(self):
        input_data = []

        for node in self.input_nodes:
            input_data.append(node.get_value())

        return input_data

    def clear_output(self):
        for node in self.output_nodes:
            node.set_value(0.0)

    def add_or_remove_random_connection(self):
        if random.random() < 0.5:
            self.add_random_connection()
        else:
            self.remove_random_connection()

    def add_or_remove_random_node(self):
        if random.random() < 0.5:
            self.add_random_node()
        else:
            self.remove_random_node()

    def remove_random_node(self):
        nodes = self.get_hidden_nodes()

        if len(nodes) > 0:
            node = random.choice(nodes)
            self.remove_node(node)

    def get_last_weight_shift_connection(self) -> Connection:
        return self.last_weight_shift_connection
