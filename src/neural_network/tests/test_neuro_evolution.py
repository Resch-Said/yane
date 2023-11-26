from src.neural_network.Genome import Genome
from src.neural_network.NeuroEvolution import NeuroEvolution


def test_get_population():
    ne = NeuroEvolution()
    genome = Genome()

    ne.get_population().add_genome(genome)

    assert ne.get_genomes_population() == [genome]


def test_pop_genome():
    ne = NeuroEvolution()
    genome1 = Genome()
    genome2 = Genome()

    genome1.set_fitness(1)
    genome2.set_fitness(2)

    ne.add_population(genome1)
    ne.add_population(genome2)

    ne.pop_genome()

    assert genome1 not in ne.get_genomes_population()
    assert genome2 in ne.get_genomes_population()


def test_remove_genome():
    ne = NeuroEvolution()
    genome1 = Genome()
    genome2 = Genome()

    genome1.set_fitness(1)
    genome2.set_fitness(2)

    ne.add_population(genome1)
    ne.add_population(genome2)

    ne.remove_genome(genome2)

    assert genome2 not in ne.get_genomes_population()
    assert genome1 in ne.get_genomes_population()


def test_get_size():
    ne = NeuroEvolution()
    genome1 = Genome()
    genome2 = Genome()

    genome1.set_fitness(1)
    genome2.set_fitness(2)

    assert ne.get_size() == 0

    ne.add_population(genome1)
    ne.add_population(genome2)

    assert ne.get_size() == 2


def test_get_best_fitness():
    ne = NeuroEvolution()
    genome1 = Genome()
    genome2 = Genome()

    genome1.set_fitness(1)
    genome2.set_fitness(2)

    ne.add_population(genome1)
    ne.add_population(genome2)

    assert ne.get_best_fitness() == 2
