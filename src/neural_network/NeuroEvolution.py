import random
import threading
from time import sleep

from src.neural_network import YaneConfig
from src.neural_network.Genome import Genome
from src.neural_network.Population import Population

yane_config = YaneConfig.load_json_config()


class NeuroEvolution:
    def __init__(self, output_neurons=1):
        self.population = Population(output_neurons=output_neurons)
        self.evaluation_queue = []
        self.ready_for_population_queue = []
        self.finished = False

    @classmethod
    def crossover(cls, genome1, genome2) -> Genome:
        return Genome.crossover(genome1, genome2)

    def get_population(self):
        return self.population

    def get_evaluation_list(self):
        return self.evaluation_queue

    def get_ready_for_population_list(self):
        return self.ready_for_population_queue

    def add_evaluation(self, genome):
        self.evaluation_queue.append(genome)

    def add_ready_for_population(self, genome):
        self.ready_for_population_queue.append(genome)

    def remove_evaluation(self, genome):
        self.evaluation_queue.remove(genome)

    def remove_ready_for_population(self, genome):
        self.ready_for_population_queue.remove(genome)

    def add_genome_evaluation(self, genome):
        self.evaluation_queue.append(genome)

    def pop_genome(self):
        self.population.pop_genome()

    def remove_genome(self, genome):
        self.population.remove_genome(genome)

    def get_genomes(self):
        return self.population.get_genomes()

    def get_size(self):
        return self.population.get_size()

    def add_output_neuron(self, neuron):
        self.population.add_output_neuron(neuron)

    # TODO: implement this method
    def train(self, min_fitness):

        # p_population_limiter -> Population
        # p_population_breeder -> Population
        # p_genome_evaluator -> Evaluation list/ready for population list
        # p_genome_integrator -> ready for population list/Population

        lock = threading.Lock()

        p_population_limiter = threading.Thread(target=self.reduce_overpopulation_multithreading, args=(lock,))
        p_population_breeder = threading.Thread(target=self.breed_population_multithreading, args=(lock,))
        p_genome_evaluator = threading.Thread(target=self.evaluate_offsprings_multithreading, args=(lock,))
        p_genome_integrator = threading.Thread(target=self.integrate_offsprings_multithreading, args=(lock,))
        p_population_condition_done = threading.Thread(target=self.check_population_condition_done_multithreading,
                                                       args=(lock, min_fitness))

        p_population_limiter.start()
        p_population_breeder.start()
        p_genome_evaluator.start()
        p_genome_integrator.start()
        p_population_condition_done.start()
        p_population_condition_done.join()

    def check_population_condition_done_multithreading(self, lock, min_fitness):
        timer = 5

        while not self.finished:
            sleep(timer)
            if self.get_size() > 0:
                lock.acquire()
                best_fitness = self.get_best_fitness()
                self.print()
                if best_fitness >= min_fitness:
                    self.finished = True
                lock.release()

    def integrate_offsprings_multithreading(self, lock):
        timer = 0

        while not self.finished:

            if len(self.get_ready_for_population_list()) > 0:
                lock.acquire()
                genome: Genome = self.get_ready_for_population_list().pop(0)
                self.population.add_genome(genome)
                lock.release()
                timer = 0
            else:
                sleep(timer)
                timer += 1

    def evaluate_offsprings_multithreading(self, lock):
        timer = 0

        while not self.finished:

            if len(self.get_evaluation_list()) > 0:
                lock.acquire()
                genome: Genome = self.get_evaluation_list().pop(0)
                lock.release()

                genome.evaluate()

                lock.acquire()
                self.add_ready_for_population(genome)
                lock.release()
                timer = 0
            else:
                sleep(timer)
                timer += 1

    def breed_population_multithreading(self, lock):

        while not self.finished:
            lock.acquire()
            genome1 = self.get_random_genome()
            genome2 = self.get_random_genome()
            lock.release()

            lock.acquire()
            child_genome = self.crossover(genome1, genome2)
            lock.release()

            child_genome.mutate()
            lock.acquire()
            self.add_evaluation(child_genome)
            lock.release()

    def reduce_overpopulation_multithreading(self, lock):
        timer = 0

        while not self.finished:
            if self.get_size() > YaneConfig.get_population_size(yane_config):
                lock.acquire()
                self.pop_genome()
                lock.release()
                timer = 0
            else:
                sleep(timer)
                timer += 1

    def get_random_genome(self):
        return random.choice(self.get_genomes())

    def print(self):
        print("Population size: " + str(self.get_size()))
        print("Average fitness: " + str(self.get_average_fitness()))
        print("Best fitness: " + str(self.get_best_fitness()))
        print("Output values: " + str(self.get_genomes()[0].get_brain().get_output_data()))

    def get_average_fitness(self):
        return self.population.get_average_fitness()

    def get_best_fitness(self):
        return self.get_genomes()[0].get_fitness()
