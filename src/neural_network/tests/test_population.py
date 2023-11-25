import math
import random

from src.neural_network.Genome import Genome
from src.neural_network.Population import Population


def test_add_genome():
    population = Population()
    assert len(population.get_genomes()) == 0

    size = 100

    for i in range(size):
        genome = Genome()
        genome.set_fitness(random.random())
        population.add_genome(genome)

    assert len(population.get_genomes()) == size

    for i in range(size - 1):
        assert population.get_genomes()[i].get_fitness() >= population.get_genomes()[i + 1].get_fitness()


def test_pop_genome():
    population = Population()
    assert len(population.get_genomes()) == 0

    size = 100

    for i in range(size):
        genome = Genome()
        genome.set_fitness(random.random())
        population.add_genome(genome)

    assert len(population.get_genomes()) == size

    top_genome = population.get_genomes()[0]

    population.pop_genome()
    population.pop_genome()
    population.pop_genome()

    assert population.get_genomes().__contains__(top_genome)

    assert len(population.get_genomes()) == size - 3


def test_remove_genome():
    population = Population()
    assert len(population.get_genomes()) == 0

    size = 100

    for i in range(size):
        genome = Genome()
        genome.set_fitness(random.random())
        population.add_genome(genome)

    assert len(population.get_genomes()) == size

    random_genome = population.get_genomes()[random.randint(0, size - 1)]

    assert population.get_genomes().__contains__(random_genome)
    population.remove_genome(random_genome)
    assert not population.get_genomes().__contains__(random_genome)

    assert len(population.get_genomes()) == size - 1


def test_get_genomes():
    population = Population()
    assert len(population.get_genomes()) == 0

    population.add_genome(Genome())
    population.add_genome(Genome())

    assert len(population.get_genomes()) == 2


def test_get_size():
    population = Population()
    assert population.get_size() == 0

    population.add_genome(Genome())
    population.add_genome(Genome())

    assert population.get_size() == 2
    population.add_genome(Genome())
    assert population.get_size() == 3


def test_get_average_fitness():
    population = Population()
    assert population.get_average_fitness() is None

    size = 100

    sum_fitness = 0
    for i in range(size):
        genome = Genome()
        genome.set_fitness(random.random())
        sum_fitness += genome.get_fitness()
        population.add_genome(genome)

    assert math.isclose(population.get_average_fitness(), sum_fitness / size)
