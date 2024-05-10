import pytest

from src.neural_network.connection.Connection import Connection
from src.neural_network.node.Node import Node
from src.neural_network.node.NodeTypes import NodeTypes
from src.neural_network.util.ActivationFunction import ActivationFunction


def input_node():
    return Node(NodeTypes.INPUT)


def hidden_node():
    return Node(NodeTypes.HIDDEN)


def output_node():
    return Node(NodeTypes.OUTPUT)


@pytest.mark.parametrize("node", [input_node(), hidden_node(), output_node()])
def test_set_value(node):
    node.set_value(5)
    assert node.get_value() == 5


@pytest.mark.parametrize("node", [input_node(), hidden_node(), output_node()])
def test_set_activation(node):
    node.set_activation('relu')
    assert node.get_activation() == 'relu'


@pytest.mark.parametrize("target_node", [input_node(), hidden_node(), output_node()])
@pytest.mark.parametrize("source_node", [input_node(), hidden_node(), output_node()])
def test_add_next_connection(source_node, target_node):
    connection = Connection(source_node, target_node, 0.5)
    source_node.add_next_connection(connection)
    expected_connection = source_node.get_next_connections().get_all_connections()[
        len(source_node.get_next_connections().get_all_connections()) - 1]

    assert expected_connection == connection


@pytest.mark.parametrize("node", [input_node(), hidden_node(), output_node()])
def test_add_previous_connection(node):
    connection = Connection(Node(NodeTypes.HIDDEN), node, 0.5)
    node.add_previous_connection(connection)
    assert node.get_previous_connections().get_all_connections()[0] == connection


@pytest.mark.parametrize("node", [input_node(), hidden_node(), output_node()])
def test_remove_next_connection(node):
    connection = Connection(node, Node(NodeTypes.HIDDEN), 0.5)
    node.add_next_connection(connection)
    node.remove_next_connection(connection)
    assert len(node.get_next_connections().get_all_connections()) == 0


@pytest.mark.parametrize("node", [input_node(), hidden_node(), output_node()])
def test_remove_previous_connection(node):
    connection = Connection(Node(NodeTypes.HIDDEN), node, 0.5)
    node.add_previous_connection(connection)
    node.remove_previous_connection(connection)
    assert len(node.get_previous_connections().get_all_connections()) == 0


@pytest.mark.parametrize("node", [input_node(), hidden_node(), output_node()])
def test_activate(node):
    node.set_value(5)
    node.activate()
    expected_value = ActivationFunction.activate(node.get_activation(), 5)

    assert node.get_value() == expected_value


@pytest.mark.parametrize("node", [input_node(), hidden_node(), output_node()])
def test_reset(node):
    node.set_value(5)
    node.reset()
    assert node.get_value() == 0


# TODO: We should just copy the connections as well
@pytest.mark.parametrize("node", [input_node(), hidden_node(), output_node()])
def test_copy(node):
    node.add_next_connection(Connection(node, Node(NodeTypes.HIDDEN), 0.5))
    node.add_previous_connection(Connection(node, Node(NodeTypes.HIDDEN), 0.3))

    copy = node.copy()
    assert copy.get_value() == node.get_value()
    assert copy.get_activation() == node.get_activation()
    assert copy.get_id() == node.get_id()
    assert copy.get_type() == node.get_type()
    assert copy != node

    assert len(copy.get_next_connections().get_all_connections()) == len(
        node.get_next_connections().get_all_connections())

    for copy_conn, orig_conn in zip(copy.get_next_connections().get_all_connections(),
                                    node.get_next_connections().get_all_connections()):
        assert copy_conn.get_weight() == orig_conn.get_weight()
        assert copy_conn.get_source_node().get_id() == orig_conn.get_source_node().get_id()
        assert copy_conn.get_target_node().get_id() == orig_conn.get_target_node().get_id()
        assert copy_conn.get_id() == orig_conn.get_id()
        assert copy_conn.get_weight_shift_direction() == orig_conn.get_weight_shift_direction()
        assert copy_conn != orig_conn

    assert len(copy.get_previous_connections().get_all_connections()) == len(
        node.get_previous_connections().get_all_connections())
    for copy_conn, orig_conn in zip(copy.get_previous_connections().get_all_connections(),
                                    node.get_previous_connections().get_all_connections()):
        assert copy_conn.get_weight() == orig_conn.get_weight()
        assert copy_conn.get_source_node().get_id() == orig_conn.get_source_node().get_id()
        assert copy_conn.get_target_node().get_id() == orig_conn.get_target_node().get_id()
        assert copy_conn.get_id() == orig_conn.get_id()
        assert copy_conn.get_weight_shift_direction() == orig_conn.get_weight_shift_direction()
        assert copy_conn != orig_conn


@pytest.mark.parametrize("node", [input_node(), hidden_node(), output_node()])
def test_fire(node, value=5):
    node.set_value(value)

    expected = 0

    if node.get_type() == NodeTypes.OUTPUT:
        expected = ActivationFunction.activate(node.get_activation(), value)

    node.fire()
    assert node.get_value() == expected


@pytest.mark.parametrize("node", [input_node(), hidden_node(), output_node()])
def test_mutate_activation_function(node):
    old_activation = node.get_activation()
    node.mutate_activation_function()
    assert node.get_activation() != old_activation
