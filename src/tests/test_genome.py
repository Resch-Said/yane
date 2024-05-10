import pytest

from src.neural_network.connection.Connection import Connection
from src.neural_network.genome.Genome import Genome
from src.neural_network.node.Node import Node
from src.neural_network.node.NodeTypes import NodeTypes


def input_node():
    return Node(NodeTypes.INPUT)


def hidden_node():
    return Node(NodeTypes.HIDDEN)


def output_node():
    return Node(NodeTypes.OUTPUT)


def test_genome_init():
    genome = Genome()
    assert genome is not None


@pytest.mark.parametrize("input_node, hidden_node, output_node", [(input_node(), hidden_node(), output_node())])
def test_get_all_connections(input_node, hidden_node, output_node):
    genome = Genome()

    genome.add_node(input_node)
    genome.add_node(hidden_node)
    genome.add_node(output_node)

    connection1 = Connection(source_node=input_node, target_node=hidden_node, weight=0.5)
    connection2 = Connection(source_node=hidden_node, target_node=output_node, weight=0.5)
    genome.add_connection(connection1)
    genome.add_connection(connection2)

    connections = genome.get_all_connections()

    assert len(connections) == 2
    assert connection1 in connections
    assert connection2 in connections


@pytest.mark.parametrize("input_node, hidden_node, output_node", [(input_node(), hidden_node(), output_node())])
def test_get_last_weight_shift_connection(input_node, hidden_node, output_node):
    genome = Genome()

    genome.add_node(input_node)
    genome.add_node(hidden_node)
    genome.add_node(output_node)

    connection1 = Connection(source_node=input_node, target_node=hidden_node, weight=0.5)
    connection2 = Connection(source_node=hidden_node, target_node=output_node, weight=0.5)
    genome.add_connection(connection1)
    genome.add_connection(connection2)

    expected = genome.mutate_weight_shift()

    last_weight_shift_connection = genome.get_last_weight_shift_connection()

    assert last_weight_shift_connection == expected


@pytest.mark.parametrize("input_node, hidden_node, output_node", [(input_node(), hidden_node(), output_node())])
def test_get_all_nodes(input_node, hidden_node, output_node):
    genome = Genome()

    genome.add_node(input_node)
    genome.add_node(hidden_node)
    genome.add_node(output_node)

    nodes = genome.get_all_nodes()

    assert len(nodes) == 3
    assert input_node in nodes
    assert hidden_node in nodes
    assert output_node in nodes


@pytest.mark.parametrize("input_node, hidden_node, output_node", [(input_node(), hidden_node(), output_node())])
def test_add_input_node(input_node, hidden_node, output_node):
    genome = Genome()
    genome.add_input_node(input_node)

    assert input_node in genome.get_input_nodes()


@pytest.mark.parametrize("input_node, hidden_node, output_node", [(input_node(), hidden_node(), output_node())])
def test_add_hidden_node(input_node, hidden_node, output_node):
    genome = Genome()

    genome.add_hidden_node(hidden_node)

    assert hidden_node in genome.get_hidden_nodes()


@pytest.mark.parametrize("input_node, hidden_node, output_node", [(input_node(), hidden_node(), output_node())])
def test_add_output_node(input_node, hidden_node, output_node):
    genome = Genome()

    genome.add_output_node(output_node)

    assert output_node in genome.get_output_nodes()


@pytest.mark.parametrize("node1", [input_node(), hidden_node(), output_node()])
@pytest.mark.parametrize("node2", [input_node(), hidden_node(), output_node()])
@pytest.mark.parametrize("node3", [input_node(), hidden_node(), output_node()])
def test_add_node(node1, node2, node3):
    genome = Genome()
    genome.add_node(node1)
    genome.add_node(node2)
    genome.add_node(node3)

    assert node1 in genome.get_all_nodes()
    assert node2 in genome.get_all_nodes()
    assert node3 in genome.get_all_nodes()


@pytest.mark.parametrize("node1", [input_node()])
@pytest.mark.parametrize("node2", [hidden_node(), output_node()])
@pytest.mark.parametrize("node3", [hidden_node(), output_node()])
def test_get_input_nodes(node1, node2, node3):
    genome = Genome()
    genome.add_node(node1)
    genome.add_node(node2)
    genome.add_node(node3)

    input_nodes = genome.get_input_nodes()

    assert node1 in input_nodes
    assert node2 not in input_nodes
    assert node3 not in input_nodes


@pytest.mark.parametrize("node1", [hidden_node()])
@pytest.mark.parametrize("node2", [input_node(), output_node()])
@pytest.mark.parametrize("node3", [input_node(), output_node()])
def test_get_hidden_nodes(node1, node2, node3):
    genome = Genome()
    genome.add_node(node1)
    genome.add_node(node2)
    genome.add_node(node3)

    hidden_nodes = genome.get_hidden_nodes()

    assert node1 in hidden_nodes
    assert node2 not in hidden_nodes
    assert node3 not in hidden_nodes


@pytest.mark.parametrize("node1", [output_node()])
@pytest.mark.parametrize("node2", [input_node(), hidden_node()])
@pytest.mark.parametrize("node3", [input_node(), hidden_node()])
def test_get_output_nodes(node1, node2, node3):
    genome = Genome()
    genome.add_node(node1)
    genome.add_node(node2)
    genome.add_node(node3)

    output_nodes = genome.get_output_nodes()

    assert node1 in output_nodes
    assert node2 not in output_nodes
    assert node3 not in output_nodes


@pytest.mark.parametrize("node1", [input_node(), hidden_node(), output_node()])
@pytest.mark.parametrize("node2", [input_node(), hidden_node(), output_node()])
@pytest.mark.parametrize("node3", [input_node(), hidden_node(), output_node()])
def test_get_node_by_id(node1, node2, node3):
    genome = Genome()
    genome.add_node(node1)
    genome.add_node(node2)
    genome.add_node(node3)

    assert genome.get_node_by_id(node1.get_id()) == node1
    assert genome.get_node_by_id(node2.get_id()) == node2
    assert genome.get_node_by_id(node3.get_id()) == node3

    assert genome.get_node_by_id(node1.get_id()) != node2
    assert genome.get_node_by_id(node1.get_id()) != node3

    assert genome.get_node_by_id(node2.get_id()) != node1
    assert genome.get_node_by_id(node2.get_id()) != node3

    assert genome.get_node_by_id(node3.get_id()) != node1
    assert genome.get_node_by_id(node3.get_id()) != node2

    assert genome.get_node_by_id(9999) is None


@pytest.mark.parametrize("node1", [input_node(), hidden_node(), output_node()])
@pytest.mark.parametrize("node2", [input_node(), hidden_node(), output_node()])
@pytest.mark.parametrize("node3", [input_node(), hidden_node(), output_node()])
def test_get_random_node(node1, node2, node3):
    genome = Genome()
    genome.add_node(node1)
    genome.add_node(node2)
    genome.add_node(node3)

    for _ in range(100):
        random_node = genome.get_random_node()
        assert random_node in [node1, node2, node3]


@pytest.mark.parametrize("node1", [input_node()])
@pytest.mark.parametrize("node2", [input_node()])
@pytest.mark.parametrize("node3", [input_node()])
def test_get_input_data(node1, node2, node3):
    genome = Genome()
    genome.add_node(node1)
    genome.add_node(node2)
    genome.add_node(node3)

    node1.set_value(1.0)
    node2.set_value(2.0)
    node3.set_value(3.0)

    input_data = genome.get_input_data()

    assert len(input_data) == 3
    assert 1.0 in input_data
    assert 2.0 in input_data
    assert 3.0 in input_data


@pytest.mark.parametrize("node1", [output_node()])
@pytest.mark.parametrize("node2", [output_node()])
@pytest.mark.parametrize("node3", [output_node()])
def test_clear_output_nodes(node1, node2, node3):
    genome = Genome()
    genome.add_node(node1)
    genome.add_node(node2)
    genome.add_node(node3)

    node1.set_value(1.0)
    node2.set_value(2.0)
    node3.set_value(3.0)

    genome.clear_output_nodes()

    assert node1.get_value() == 0.0
    assert node2.get_value() == 0.0
    assert node3.get_value() == 0.0


@pytest.mark.parametrize("input_node1", [input_node()])
@pytest.mark.parametrize("input_node2", [input_node()])
@pytest.mark.parametrize("input_node3", [input_node()])
@pytest.mark.parametrize("hidden_node1", [hidden_node()])
@pytest.mark.parametrize("hidden_node2", [hidden_node()])
@pytest.mark.parametrize("output_node1", [output_node()])
@pytest.mark.parametrize("output_node2", [output_node()])
def test_get_relevant_input_nodes(input_node1, input_node2, input_node3, hidden_node1, hidden_node2, output_node1,
                                  output_node2):
    genome = Genome()
    genome.add_node(input_node1)
    genome.add_node(input_node2)
    genome.add_node(input_node3)
    genome.add_node(hidden_node1)
    genome.add_node(hidden_node2)
    genome.add_node(output_node1)
    genome.add_node(output_node2)

    # input_node1 -> hidden_node1 -> output_node1
    # input_node2 -> hidden_node2
    # input_node3 -> output_node2
    # Relevant input nodes are input_node1 and input_node3
    connection1 = Connection(source_node=input_node1, target_node=hidden_node1, weight=0.5)
    connection2 = Connection(source_node=hidden_node1, target_node=output_node1, weight=0.5)
    connection3 = Connection(source_node=input_node2, target_node=hidden_node2, weight=0.5)
    connection4 = Connection(source_node=input_node3, target_node=output_node2, weight=0.5)

    genome.add_connection(connection1)
    genome.add_connection(connection2)
    genome.add_connection(connection3)
    genome.add_connection(connection4)

    relevant_input_nodes = genome.get_relevant_input_nodes()

    assert len(relevant_input_nodes) == 2
    assert input_node1 in relevant_input_nodes
    assert input_node2 not in relevant_input_nodes
    assert input_node3 in relevant_input_nodes


@pytest.mark.parametrize("node1", [output_node()])
@pytest.mark.parametrize("node2", [output_node()])
@pytest.mark.parametrize("node3", [output_node()])
def test_get_output_data(node1, node2, node3):
    genome = Genome()
    genome.add_node(node1)
    genome.add_node(node2)
    genome.add_node(node3)

    node1.set_value(1.0)
    node2.set_value(2.0)
    node3.set_value(3.0)

    output_data = genome.get_output_data()

    assert len(output_data) == 3
    assert 1.0 in output_data
    assert 2.0 in output_data
    assert 3.0 in output_data


@pytest.mark.parametrize("input_node", [input_node()])
@pytest.mark.parametrize("output_node", [output_node()])
def test_add_random_node(input_node, output_node):
    genome = Genome()
    genome.add_node(input_node)
    genome.add_node(output_node)

    connection = Connection(source_node=input_node, target_node=output_node, weight=0.5)
    genome.add_connection(connection)

    genome.add_random_node()

    all_nodes = genome.get_all_nodes()
    all_connections = genome.get_all_connections()

    assert len(all_nodes) == 3  # input_node, output_node, and the new node
    assert len(all_connections) == 2  # the original connection and the new connection

    new_node = None
    for node in all_nodes:
        if node != input_node and node != output_node:
            new_node = node
            break

    assert new_node is not None

    new_connections = [conn for conn in all_connections if conn != connection]
    assert len(new_connections) == 2

    assert any(conn.source_node == input_node and conn.target_node == new_node for conn in new_connections)
    assert any(conn.source_node == new_node and conn.target_node == output_node for conn in new_connections)


@pytest.mark.parametrize("input_node1", [input_node()])
@pytest.mark.parametrize("input_node2", [input_node()])
@pytest.mark.parametrize("hidden_node1", [hidden_node()])
@pytest.mark.parametrize("hidden_node2", [hidden_node()])
@pytest.mark.parametrize("output_node1", [output_node()])
@pytest.mark.parametrize("output_node2", [output_node()])
def test_remove_random_node(input_node1, input_node2, hidden_node1, hidden_node2, output_node1, output_node2):
    genome = Genome()
    genome.add_node(input_node1)
    genome.add_node(input_node2)
    genome.add_node(hidden_node1)
    genome.add_node(hidden_node2)
    genome.add_node(output_node1)
    genome.add_node(output_node2)

    # input_node1 -> hidden_node1 -> output_node1
    # input_node2 -> hidden_node2 -> output_node2
    connection1 = Connection(source_node=input_node1, target_node=hidden_node1, weight=0.5)
    connection2 = Connection(source_node=hidden_node1, target_node=output_node1, weight=0.5)
    connection3 = Connection(source_node=input_node2, target_node=hidden_node2, weight=0.5)
    connection4 = Connection(source_node=hidden_node2, target_node=output_node2, weight=0.5)

    genome.add_connection(connection1)
    genome.add_connection(connection2)
    genome.add_connection(connection3)
    genome.add_connection(connection4)

    initial_node_count = len(genome.get_all_nodes())
    initial_connection_count = len(genome.get_all_connections())

    genome.remove_random_node()

    assert len(genome.get_all_nodes()) == initial_node_count - 1
    assert len(genome.get_all_connections()) <= initial_connection_count - 2


@pytest.mark.parametrize("input_node1", [input_node()])
@pytest.mark.parametrize("hidden_node1", [hidden_node()])
@pytest.mark.parametrize("output_node1", [output_node()])
def test_remove_node(input_node1, hidden_node1, output_node1):
    genome = Genome()
    genome.add_node(input_node1)
    genome.add_node(hidden_node1)
    genome.add_node(output_node1)

    connection1 = Connection(source_node=input_node1, target_node=hidden_node1, weight=0.5)
    connection2 = Connection(source_node=hidden_node1, target_node=output_node1, weight=0.5)

    genome.add_connection(connection1)
    genome.add_connection(connection2)

    initial_node_count = len(genome.get_all_nodes())
    initial_connection_count = len(genome.get_all_connections())

    genome.remove_node(hidden_node1)

    assert len(genome.get_all_nodes()) == initial_node_count - 1
    assert len(genome.get_all_connections()) == initial_connection_count - 2
    assert hidden_node1 not in genome.get_all_nodes()
    assert connection1 not in genome.get_all_connections()
    assert connection2 not in genome.get_all_connections()


def test_set_number_of_outputs():
    genome = Genome()

    initial_output_node_count = len(genome.get_output_nodes())

    genome.set_number_of_outputs(5)

    final_output_node_count = len(genome.get_output_nodes())

    assert final_output_node_count - initial_output_node_count == 5


@pytest.mark.parametrize("node1", [hidden_node(), output_node()])
@pytest.mark.parametrize("node2", [hidden_node(), output_node()])
@pytest.mark.parametrize("node3", [hidden_node(), output_node()])
def test_clear_hidden_output_nodes(node1, node2, node3):
    genome = Genome()
    genome.add_node(node1)
    genome.add_node(node2)
    genome.add_node(node3)

    node1.set_value(1.0)
    node2.set_value(2.0)
    node3.set_value(3.0)

    genome.clear_hidden_output_nodes()

    assert node1.get_value() == 0.0
    assert node2.get_value() == 0.0
    assert node3.get_value() == 0.0


@pytest.mark.parametrize("input_node1", [input_node()])
@pytest.mark.parametrize("input_node2", [input_node()])
@pytest.mark.parametrize("input_node3", [input_node()])
@pytest.mark.parametrize("hidden_node1", [hidden_node()])
@pytest.mark.parametrize("hidden_node2", [hidden_node()])
@pytest.mark.parametrize("output_node1", [output_node()])
@pytest.mark.parametrize("output_node2", [output_node()])
def test_set_input_data(input_node1, input_node2, input_node3, hidden_node1, hidden_node2, output_node1, output_node2):
    genome = Genome()
    genome.add_node(input_node1)
    genome.add_node(input_node2)
    genome.add_node(input_node3)
    genome.add_node(hidden_node1)
    genome.add_node(hidden_node2)
    genome.add_node(output_node1)
    genome.add_node(output_node2)

    input_node1.input_pos = 0
    input_node2.input_pos = 1
    input_node3.input_pos = 2

    # input_node1 -> hidden_node1 -> output_node1
    # input_node2 -> hidden_node2 -> output_node2
    connection1 = Connection(source_node=input_node1, target_node=hidden_node1, weight=0.5)
    connection2 = Connection(source_node=hidden_node1, target_node=output_node1, weight=0.5)
    connection3 = Connection(source_node=input_node2, target_node=hidden_node2, weight=0.5)
    connection4 = Connection(source_node=hidden_node2, target_node=output_node2, weight=0.5)

    genome.add_connection(connection1)
    genome.add_connection(connection2)
    genome.add_connection(connection3)
    genome.add_connection(connection4)

    input_data = [1.0, 2.0, 3.0]
    relevant_input_nodes = genome.set_input_data(input_data)

    assert len(relevant_input_nodes) == 2
    assert input_node1 in relevant_input_nodes
    assert input_node2 in relevant_input_nodes
    assert input_node3 not in relevant_input_nodes
    assert input_node1.get_value() == 1.0
    assert input_node2.get_value() == 2.0
    assert input_node3.get_value() == 0.0  # input_node3 is not relevant

    next_trigger_nodes = genome.next_trigger_nodes
    assert len(next_trigger_nodes) == 2
    assert input_node1 in next_trigger_nodes
    assert input_node2 in next_trigger_nodes

    connection5 = Connection(source_node=input_node3, target_node=hidden_node2, weight=0.5)
    genome.add_connection(connection5)

    input_data = [3.0, 4.0, 5.0]
    relevant_input_nodes = genome.set_input_data(input_data)

    assert len(relevant_input_nodes) == 2
    assert input_node1 in relevant_input_nodes
    assert input_node2 in relevant_input_nodes
    assert input_node1.get_value() == 3.0
    assert input_node2.get_value() == 4.0
    assert input_node3.get_value() == 0.0  # Still not relevant because a genomes structure is static.

    input_data = [3.0]
    relevant_input_nodes = genome.set_input_data(input_data)

    assert len(relevant_input_nodes) == 2
    assert input_node1 in relevant_input_nodes
    assert input_node2 in relevant_input_nodes
    assert input_node3 not in relevant_input_nodes
    assert input_node1.get_value() == 3.0
    assert input_node2.get_value() == 0.0
    assert input_node3.get_value() == 0.0


@pytest.mark.skip(reason="Not implemented")
def test_mutate():
    pass


@pytest.mark.skip(reason="Not implemented")
def test_mutate_nodes():
    pass


@pytest.mark.skip(reason="Not implemented")
def test_mutate_connections():
    pass


@pytest.mark.skip(reason="Not implemented")
def test_mutate_mutation_rates():
    pass


@pytest.mark.skip(reason="Not implemented")
def test_mutate_mutation_nums():
    pass


@pytest.mark.skip(reason="Not implemented")
def test_evaluate():
    pass


@pytest.mark.parametrize("input_node1", [input_node()])
@pytest.mark.parametrize("hidden_node1", [hidden_node()])
@pytest.mark.parametrize("output_node1", [output_node()])
def test_remove_random_connection(input_node1, hidden_node1, output_node1):
    genome = Genome()
    genome.add_node(input_node1)
    genome.add_node(hidden_node1)
    genome.add_node(output_node1)

    connection1 = Connection(source_node=input_node1, target_node=hidden_node1, weight=0.5)
    connection2 = Connection(source_node=hidden_node1, target_node=output_node1, weight=0.5)

    genome.add_connection(connection1)
    genome.add_connection(connection2)

    initial_connection_count = len(genome.get_all_connections())

    genome.remove_random_connection()

    assert len(genome.get_all_connections()) == initial_connection_count - 1


@pytest.mark.parametrize("input_node", [input_node()])
@pytest.mark.parametrize("hidden_node", [hidden_node()])
@pytest.mark.parametrize("output_node", [output_node()])
def test_remove_connection(input_node, hidden_node, output_node):
    genome = Genome()
    genome.add_node(input_node)
    genome.add_node(hidden_node)
    genome.add_node(output_node)

    connection = Connection(source_node=input_node, target_node=hidden_node, weight=0.5)
    genome.add_connection(connection)

    initial_connection_count = len(genome.get_all_connections())

    genome.remove_connection(connection)

    assert len(genome.get_all_connections()) == initial_connection_count - 1
    assert connection not in genome.get_all_connections()


@pytest.mark.parametrize("node1", [input_node(), hidden_node(), output_node()])
@pytest.mark.parametrize("node2", [input_node(), hidden_node(), output_node()])
@pytest.mark.parametrize("node3", [input_node(), hidden_node(), output_node()])
def test_add_random_connection(node1, node2, node3):
    genome = Genome()
    genome.add_node(node1)
    genome.add_node(node2)
    genome.add_node(node3)

    initial_connection_count = len(genome.get_all_connections())

    new_connection = genome.add_random_connection()

    if new_connection is None:
        assert len(genome.get_all_connections()) == initial_connection_count
    else:
        assert len(genome.get_all_connections()) == initial_connection_count + 1
        assert new_connection.get_source_node() != new_connection.get_target_node()


@pytest.mark.parametrize("input_node", [input_node()])
@pytest.mark.parametrize("hidden_node", [hidden_node()])
@pytest.mark.parametrize("output_node", [output_node()])
def test_add_connection(input_node, hidden_node, output_node):
    genome = Genome()
    genome.add_node(input_node)
    genome.add_node(hidden_node)
    genome.add_node(output_node)

    initial_connection_count = len(genome.get_all_connections())

    connection = Connection(source_node=input_node, target_node=hidden_node, weight=0.5)
    genome.add_connection(connection)

    assert len(genome.get_all_connections()) == initial_connection_count + 1
    assert connection in genome.get_all_connections()


@pytest.mark.skip(reason="Not implemented")
def test_crossover_connections():
    pass


@pytest.mark.skip(reason="Not implemented")
def test_align_gene_ids():
    pass


@pytest.mark.skip(reason="Not implemented")
def test_crossover_genes():
    pass


@pytest.mark.skip(reason="Not implemented")
def test_crossover_nodes():
    pass


@pytest.mark.skip(reason="Not implemented")
def test_crossover():
    pass


def test_fitness():
    genome = Genome()

    genome.set_fitness(0.75)

    assert genome.get_fitness() == 0.75


@pytest.mark.parametrize("input_node1", [input_node()])
@pytest.mark.parametrize("hidden_node1", [hidden_node()])
@pytest.mark.parametrize("output_node1", [output_node()])
def test_net_cost(input_node1, hidden_node1, output_node1):
    genome = Genome()
    genome.add_node(input_node1)
    genome.add_node(hidden_node1)
    genome.add_node(output_node1)

    connection1 = Connection(source_node=input_node1, target_node=hidden_node1, weight=0.5)
    connection2 = Connection(source_node=hidden_node1, target_node=output_node1, weight=0.5)

    genome.add_connection(connection1)
    genome.add_connection(connection2)

    net_cost = genome.calculate_net_cost()

    assert net_cost == len(genome.get_all_nodes()) + len(genome.get_all_connections())


@pytest.mark.skip(reason="Not implemented")
def test_tick():
    pass


@pytest.mark.skip(reason="Not implemented")
def test_forward_propagation():
    pass


@pytest.mark.skip(reason="Not implemented")
def test_copy():
    pass


def test_parent():
    parent_genome = Genome()
    child_genome = Genome()

    child_genome.set_parent(parent_genome)

    assert child_genome.get_parent() == parent_genome


def test_reproduction_count():
    genome = Genome()

    genome.set_reproduction_count(5)

    assert genome.get_reproduction_count() == 5


def test_bad_reproduction_count():
    genome = Genome()

    genome.set_bad_reproduction_count(3)

    assert genome.get_bad_reproduction_count() == 3


@pytest.mark.skip(reason="Not implemented")
def test_species_compatibility():
    pass


@pytest.mark.parametrize("input_node", [input_node()])
@pytest.mark.parametrize("hidden_node", [hidden_node()])
@pytest.mark.parametrize("output_node", [output_node()])
def test_average_weight(input_node, hidden_node, output_node):
    genome = Genome()
    genome.add_node(input_node)
    genome.add_node(hidden_node)
    genome.add_node(output_node)

    connection1 = Connection(source_node=input_node, target_node=hidden_node, weight=0.5)
    connection2 = Connection(source_node=hidden_node, target_node=output_node, weight=1.5)

    genome.add_connection(connection1)
    genome.add_connection(connection2)

    average_weight = genome.get_average_weight()

    assert average_weight == (0.5 + 1.5) / 2


@pytest.mark.parametrize("input_node", [input_node()])
@pytest.mark.parametrize("hidden_node", [hidden_node()])
def test_has_connection(input_node, hidden_node):
    genome = Genome()
    genome.add_node(input_node)
    genome.add_node(hidden_node)

    connection = Connection(source_node=input_node, target_node=hidden_node, weight=0.5)
    genome.add_connection(connection)

    assert genome.has_connection(connection) == True

    genome.remove_connection(connection)

    assert genome.has_connection(connection) == False

    non_existent_connection = Connection(source_node=hidden_node, target_node=input_node, weight=0.5)
    assert genome.has_connection(non_existent_connection) == False
