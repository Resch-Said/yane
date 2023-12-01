from src.neural_network.HiddenNeuron import HiddenNeuron
from src.neural_network.InputNeuron import InputNeuron

from src.neural_network.Connection import Connection
from src.neural_network.Node import Node


def test_set_weight():
    con = Connection()
    con.set_weight(0.5)
    assert con.get_weight() == 0.5


def test_set_in_neuron():
    con = Connection()
    neuron = InputNeuron()
    con.set_in_node(neuron)

    assert con.get_in_node() == neuron


def test_set_out_neuron():
    con = Connection()
    neuron = Node()
    con.set_out_node(neuron)

    assert con.get_out_node() == neuron


def test_set_in_out_neuron():
    con = Connection()
    in_neuron = InputNeuron()
    out_neuron = Node()
    con.set_in_node(in_neuron)
    con.set_out_node(out_neuron)

    assert con.get_in_node() == in_neuron
    assert con.get_out_node() == out_neuron


def test_set_enabled():
    con = Connection()
    con.set_enabled(False)
    assert not con.is_enabled()


def test_is_enabled():
    con = Connection()
    con.set_enabled(True)
    assert con.is_enabled()


def test_get_id():
    Connection.ID = 0

    con = Connection()
    assert con.get_id() == 0
    con = Connection()
    assert con.get_id() == 1

    con2 = None

    for i in range(100):
        con2 = Connection()
    assert con2.get_id() == 101
    assert con.get_id() == 1


def test_copy():
    con = Connection()
    con.set_weight(0.5)
    con.set_enabled(False)

    new_con = con.copy()

    assert new_con.get_weight() == con.get_weight()
    assert new_con.is_enabled() == con.is_enabled()
    assert new_con.get_id() == con.get_id()
    assert new_con.get_in_node() == con.get_in_node()
    assert new_con.get_out_node() == con.get_out_node()
    assert new_con is not con
    assert id(new_con) != id(con)


def test_copy_2():
    con = Connection()
    con.set_weight(0.5)
    con.set_enabled(False)

    new_con = con.copy()

    neuron_in = InputNeuron()
    neuron_out = HiddenNeuron()

    new_con.set_in_node(neuron_in)
    new_con.set_out_node(neuron_out)

    assert new_con.get_in_node() == neuron_in
    assert new_con.get_out_node() == neuron_out
    assert con.get_in_node() != neuron_in
    assert con.get_out_node() != neuron_out


def test_mutate_weight():
    con = Connection()
    con.set_weight(0.5)
    con.mutate_weight_random()

    assert con.get_weight() != 0.5


def test_mutate_enabled():
    con = Connection()
    con.set_enabled(True)
    con.mutate_enabled()

    assert not con.is_enabled()


def test_mutate_weight_shift():
    con = Connection()
    con.set_weight(0.5)
    con.mutate_weight_shift()

    assert con.get_weight() != 0.5
