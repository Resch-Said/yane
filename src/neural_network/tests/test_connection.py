from src.neural_network.Connection import Connection
from src.neural_network.Neuron import Neuron

neuron1 = Neuron()
neuron2 = Neuron()


def test_init():
    connection = Connection(neuron1, neuron2, 1)
    assert connection.neuron_from == neuron1
    assert connection.neuron_to == neuron2
    assert connection.weight == 1
