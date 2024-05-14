from neural_network.population import Population
from neural_network.genome import Genome


class Yane:
    def __init__(self):
        self.min_fitness = None
        self.population: Population = Population(max_population=100)
        self.selected_candidate: Genome = self._select_candidate()

    def set_min_fitness(self, min_fitness: float):
        """Stop condition for the genetic algorithm

        Args:
            min_fitness (float): After reaching this fitness, the genetic algorithm stops
        """
        self.min_fitness = min_fitness

    def step(self, input_data) -> list[float]:
        """current selected neural network does one step and returns array of output values

        Args:
            input_data (array of input values): input values to insert into the current selected neural network
        """
        return self.selected_candidate.tick(input_data)

    def run(self, fitness):
        """runs the genetic algorithm with the given fitness function until the stop condition is reached

        Args:
            fitness (Fitness function which should return a fitness value)
        """
        done = False
        while not done:
            self.selected_candidate = self._select_candidate()
            fitness_value = fitness()
            self.population.move_genome_to_evaluated(
                self.selected_candidate)

            if fitness_value >= self.min_fitness:
                done = True

    def get_best_solution(self) -> 'Genome':
        """returns the best solution found by the genetic algorithm
        """
        return self.population.get_best_solution()

    def print(self):
        pass

    def _select_candidate(self):
        return self.population.select_candidate(evaluated=False, unevaluated=True)


def main():
    yane = Yane()
    yane.set_min_fitness(500)
    fitness_value = 0

    def fitness():
        nonlocal fitness_value
        fitness_value += 1
        return fitness_value

    yane.run(fitness)


if __name__ == '__main__':
    main()
