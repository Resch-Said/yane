import random

from src.neural_network.connection import Connection
from src.neural_network.node import Node


class ConnectionCollection:
    def __init__(self):
        self.target_nodes = {}
        self.keys = []
        self.key_to_index = {}
        self.id_to_connection = {}

    def add(self, connection: Connection):
        target_node = connection.get_target_node()
        if target_node not in self.target_nodes:
            self.keys.append(target_node)
            self.key_to_index[target_node] = len(self.keys) - 1
        self.target_nodes[target_node] = connection
        self.id_to_connection[connection.get_id()] = connection

    def remove(self, connection: Connection):
        target_node = connection.get_target_node()
        if target_node in self.target_nodes and self.target_nodes[target_node] == connection:
            del self.target_nodes[target_node]
            del self.id_to_connection[connection.get_id()]
            self._remove_key(target_node)

    def _remove_key(self, key):
        key_index = self.key_to_index[key]
        last_key = self.keys[-1]

        self.keys[key_index] = last_key
        self.key_to_index[last_key] = key_index

        self.keys.pop()
        del self.key_to_index[key]

    def get_random_key(self):
        return random.choice(self.keys)

    def get_connection(self, target_node: Node) -> Connection:
        if target_node in self.target_nodes:
            return self.target_nodes[target_node]
        return None

    def get_all_connections(self) -> list[Connection]:
        connections = []
        for target_node in self.target_nodes:
            connections.append(self.target_nodes[target_node])
        return connections

    def get_all_target_nodes(self) -> list[Node]:
        return self.keys

    def __contains__(self, target_node: Node):
        return target_node in self.target_nodes

    def __iter__(self):
        for connections in self.target_nodes.values():
            yield from connections

    def copy(self) -> 'ConnectionCollection':
        new_collection = ConnectionCollection()
        for connection in self.get_all_connections():
            new_collection.add(connection.copy())
        return new_collection

    def remove_by_id(self, connection_id):
        if connection_id in self.id_to_connection:
            self.remove(self.id_to_connection[connection_id])
