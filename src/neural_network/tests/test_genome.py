from copy import deepcopy

from src.neural_network.Genome import Genome
from src.neural_network.HiddenNeuron import HiddenNeuron
from src.neural_network.InputNeuron import InputNeuron
from src.neural_network.OutputNeuron import OutputNeuron


class GeneDummyTest:
    ID = 0

    def __init__(self, value):
        self.id = GeneDummyTest.ID
        self.value = value
        GeneDummyTest.ID += 1

    def get_id(self):
        return self.id

    def get_value(self):
        return self.value

    def set_value(self, value):
        self.value = value

    def copy(self):
        return deepcopy(self)


def test_crossover_neurons():
    input1 = InputNeuron()
    hidden1 = HiddenNeuron()
    out1 = OutputNeuron()

    genome1 = Genome()
    genome1.add_neuron(input1)
    genome1.add_neuron(out1)

    genome2 = Genome.crossover(genome1, genome1)

    genome2.print()

    assert genome1 != genome2
    assert genome1.get_brain().get_all_neurons() != genome2.get_brain().get_all_neurons()
    assert len(genome1.get_brain().get_all_neurons()) == len(genome2.get_brain().get_all_neurons())
    assert genome1.get_brain().get_all_neurons()[0].get_id() == genome2.get_brain().get_all_neurons()[0].get_id()
    assert genome1.get_brain().get_all_neurons()[1].get_id() == genome2.get_brain().get_all_neurons()[1].get_id()
    assert genome1.get_brain().get_all_neurons()[1] != genome2.get_brain().get_all_neurons()[1]


def test_crossover_neurons2():
    input1 = InputNeuron()
    hidden1 = HiddenNeuron()
    hidden2 = HiddenNeuron()
    out1 = OutputNeuron()

    genome1 = Genome()
    genome1.add_neuron(input1)
    genome1.add_neuron(out1)

    genome2 = Genome.crossover(genome1, genome1)

    genome1.add_neuron(hidden1)
    genome2.add_neuron(hidden2)
    genome2.get_brain().add_random_connection()
    genome1.get_brain().add_random_connection()

    genome3 = Genome.crossover(genome1, genome2)

    assert genome1 != genome3

    assert genome1.get_brain().get_all_neurons() != genome3.get_brain().get_all_neurons()
    assert len(genome1.get_brain().get_all_neurons()) + 1 == len(genome3.get_brain().get_all_neurons())
    assert len(genome2.get_brain().get_all_neurons()) != len(genome3.get_brain().get_all_neurons())
    assert genome1.get_brain().get_all_neurons()[0].get_id() == genome3.get_brain().get_all_neurons()[0].get_id()
    assert genome1.get_brain().get_all_neurons()[1].get_id() == genome3.get_brain().get_all_neurons()[1].get_id()
    assert genome1.get_brain().get_all_neurons()[1] != genome3.get_brain().get_all_neurons()[1]
    assert len(genome3.get_brain().get_all_neurons()) == 4
    assert len(genome1.get_brain().get_all_neurons()) == 3
    assert len(genome2.get_brain().get_all_neurons()) == 3
    assert len(genome3.get_brain().get_all_connections()) in [1, 2]


def test_crossover_genes():
    print()

    def print_gene(genes):
        for gene in genes:
            print("Gene: " + str(gene.get_id()) + " Value: " + str(gene.get_value()))

    genes1 = [GeneDummyTest(2), GeneDummyTest(2), GeneDummyTest(2), GeneDummyTest(2), GeneDummyTest(2)]
    genes2 = deepcopy(genes1)
    genes2[0].set_value(1)
    genes2[1].set_value(1)
    genes2[2].set_value(1)
    genes2[2].id = 5
    genes2[3].set_value(1)
    genes2[4].set_value(1)

    genes3 = Genome.crossover_genes(genes1, genes2)

    print_gene(genes3)

    assert genes1 != genes3
    assert len(genes1) == len(genes2)
    for i in range(len(genes1)):
        assert genes1[i].get_id() == genes3[i].get_id()
        if genes1[i].get_value() == genes3[i].get_value():
            assert True
        elif genes2[i].get_value() == genes3[i].get_value():
            assert True
        else:
            assert False

    assert genes3[5].get_id() == 5


def test_crossover_genes_no_duplications():
    print()

    def print_gene(genes):
        for gene in genes:
            print("Gene: " + str(gene.get_id()) + " Value: " + str(gene.get_value()))

    genes1 = [GeneDummyTest(2), GeneDummyTest(2), GeneDummyTest(2), GeneDummyTest(2), GeneDummyTest(2)]
    genes2 = deepcopy(genes1)
    genes2[0].set_value(1)
    genes2[1].set_value(1)
    genes2[2].set_value(1)
    genes2[2].id = 5
    genes2[3].set_value(1)
    genes2[4].set_value(1)

    genes3 = Genome.crossover_genes(genes1, genes2)

    for i in range(len(genes3)):
        for j in range(len(genes3)):
            if i != j:
                assert genes3[i].get_id() != genes3[j].get_id()


def test_evaluate():
    assert False


def test_copy():
    genome = Genome()
    genome2 = genome.copy()
    assert genome != genome2
