import pytest

from src.neural_network.connection.Connection import Connection
from src.neural_network.node.Node import Node
from src.neural_network.node.NodeTypes import NodeTypes


def input_node():
    return Node(NodeTypes.INPUT)


def hidden_node():
    return Node(NodeTypes.HIDDEN)


def output_node():
    return Node(NodeTypes.OUTPUT)


def test_initial_weight():
    connection = Connection(input_node(), hidden_node(), 0)
    assert connection.get_weight() == 0


def test_update_weight():
    connection = Connection(input_node(), hidden_node())
    connection.set_weight(0.5)
    assert connection.get_weight() == 0.5


def test_reset_weight():
    connection = Connection(input_node(), hidden_node())
    connection.set_weight(0.5)
    connection.set_weight(0)
    assert connection.get_weight() == 0


@pytest.mark.parametrize("node_source", [input_node(), hidden_node(), output_node()])
def test_set_source_node(node_source):
    connection = Connection()
    connection.set_source_node(node_source)
    assert connection.get_source_node() == node_source


@pytest.mark.parametrize("node_target", [input_node(), hidden_node(), output_node()])
def test_set_target_node(node_target):
    connection = Connection()
    connection.set_target_node(node_target)
    assert connection.get_target_node() == node_target


def test_get_id():
    connection = Connection()
    assert connection.get_id() == Connection.global_connection_id - 1


@pytest.mark.parametrize("node_target", [input_node(), hidden_node(), output_node()])
@pytest.mark.parametrize("node_source", [input_node(), hidden_node(), output_node()])
def test_copy(node_target, node_source):
    connection = Connection(node_source, node_target, 0.5, True, 1)
    new_connection = connection.copy()
    assert new_connection.get_source_node() == node_source
    assert new_connection.get_target_node() == node_target
    assert new_connection.get_weight() == 0.5
    assert new_connection.get_id() == 1
    assert new_connection.get_weight_shift_direction() == True


def test_mutate_weight_random():
    connection = Connection(input_node(), hidden_node(), 0.5, True, 1)
    connection.mutate_weight_random()
    assert connection.get_weight() != 0.5


def test_get_weight_shift_direction():
    connection = Connection(input_node(), hidden_node(), 0.5, True, 1)
    assert connection.get_weight_shift_direction() == True


def test_set_weight_shift_direction():
    connection = Connection(input_node(), hidden_node(), 0.5, True, 1)
    connection.switch_weight_shift_direction()
    assert connection.get_weight_shift_direction() == False


@pytest.mark.parametrize("node_target", [input_node(), hidden_node(), output_node()])
@pytest.mark.parametrize("node_source", [input_node(), hidden_node(), output_node()])
def test_get_source_node(node_target, node_source):
    connection = Connection(node_source, node_target, 0.5, True, 1)
    assert connection.get_source_node() == node_source


@pytest.mark.parametrize("node_target", [input_node(), hidden_node(), output_node()])
@pytest.mark.parametrize("node_source", [input_node(), hidden_node(), output_node()])
def test_get_target_node(node_target, node_source):
    connection = Connection(node_source, node_target, 0.5, True, 1)
    assert connection.get_target_node() == node_target


def test_get_weight():
    connection = Connection(input_node(), hidden_node(), 0.5, True, 1)
    assert connection.get_weight() == 0.5
