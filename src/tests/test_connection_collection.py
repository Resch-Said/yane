import pytest

from src.neural_network.connection.Connection import Connection
from src.neural_network.connection.ConnectionCollection import ConnectionCollection
from src.neural_network.node.Node import Node
from src.neural_network.node.NodeTypes import NodeTypes


def input_node():
    return Node(NodeTypes.INPUT)


def hidden_node():
    return Node(NodeTypes.HIDDEN)


def output_node():
    return Node(NodeTypes.OUTPUT)


@pytest.mark.parametrize("node_target", [input_node(), hidden_node(), output_node()])
@pytest.mark.parametrize("node_source", [input_node(), hidden_node(), output_node()])
def test_add(node_target, node_source):
    collection = ConnectionCollection()
    connection = Connection(node_source, node_target, 0.5)
    collection.add(connection)

    assert connection.get_target_node() in collection.target_nodes
    assert connection.get_target_node() in collection.keys
    assert collection.target_nodes[connection.get_target_node()] == connection
    assert collection.keys[collection.key_to_index[connection.get_target_node()]] == connection.get_target_node()
    assert collection.key_to_index[connection.get_target_node()] == len(collection.keys) - 1
    assert len(collection.keys) == len(collection.key_to_index)
    assert len(collection.keys) == len(collection.target_nodes)

    assert connection.get_id() in collection.id_to_connection
    assert collection.id_to_connection[connection.get_id()] == connection


@pytest.mark.parametrize("node_target", [input_node(), hidden_node(), output_node()])
@pytest.mark.parametrize("node_source", [input_node(), hidden_node(), output_node()])
def test_remove(node_target, node_source):
    collection = ConnectionCollection()
    connection = Connection(node_source, node_target, 0.5)
    collection.add(connection)
    collection.remove(connection)

    assert connection.get_target_node() not in collection.target_nodes
    assert connection.get_target_node() not in collection.keys
    assert connection.get_target_node() not in collection.key_to_index
    assert len(collection.keys) == len(collection.key_to_index)
    assert len(collection.keys) == len(collection.target_nodes)

    assert connection.get_id() not in collection.id_to_connection


@pytest.mark.parametrize("node_target", [input_node(), hidden_node(), output_node()])
@pytest.mark.parametrize("node_source", [input_node(), hidden_node(), output_node()])
def test_get_connection(node_target, node_source):
    collection = ConnectionCollection()
    connection = Connection(node_source, node_target, 0.5)
    collection.add(connection)
    assert collection.get_connection(connection.get_target_node()) == connection


@pytest.mark.parametrize("node_target", [input_node(), hidden_node(), output_node()])
@pytest.mark.parametrize("node_source", [input_node(), hidden_node(), output_node()])
def test_get_all_connections(node_target, node_source):
    collection = ConnectionCollection()
    connections = [Connection(node_source, hidden_node(), 0.5) for _ in range(5)]
    for connection in connections:
        collection.add(connection)
    assert set(collection.get_all_connections()) == set(connections)


@pytest.mark.parametrize("node_target", [input_node(), hidden_node(), output_node()])
@pytest.mark.parametrize("node_source", [input_node(), hidden_node(), output_node()])
def test_get_all_target_nodes(node_target, node_source):
    collection = ConnectionCollection()
    connections = [Connection(node_source, hidden_node(), 0.5) for _ in range(5)]
    for connection in connections:
        collection.add(connection)
    assert set(collection.get_all_target_nodes()) == set([connection.get_target_node() for connection in connections])


@pytest.mark.parametrize("node_target", [input_node(), hidden_node(), output_node()])
@pytest.mark.parametrize("node_source", [input_node(), hidden_node(), output_node()])
def test_copy(node_target, node_source):
    collection = ConnectionCollection()
    connections = [Connection(node_source, hidden_node(), 0.5) for _ in range(5)]
    for connection in connections:
        collection.add(connection)
    copy = collection.copy()

    assert set(connection.get_id() for connection in connections) == set(
        connection.get_id() for connection in collection.get_all_connections())
    assert set(connection.get_id() for connection in connections) == set(
        connection.get_id() for connection in copy.get_all_connections())


@pytest.mark.parametrize("node_target", [input_node(), hidden_node(), output_node()])
@pytest.mark.parametrize("node_source", [input_node(), hidden_node(), output_node()])
def test_remove_by_id(node_target, node_source):
    collection = ConnectionCollection()
    connection = Connection(node_source, node_target, 0.5)
    collection.add(connection)

    assert connection.get_id() in collection.id_to_connection

    collection.remove_by_id(connection.get_id())

    assert connection.get_id() not in collection.id_to_connection
