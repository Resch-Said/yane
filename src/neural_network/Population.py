import math
import random

from neural_network.genome import Genome


class Population:
    def __init__(self, max_population) -> None:
        self.max_population = max_population
        self.length = 1
        self._genomes_unevaluated = [Genome()]
        self._genomes_evaluated = []

    def select_candidate(self, evaluated, unevaluated) -> Genome:
        if len(self._genomes_unevaluated) == 0:
            self._create_genome()

        if self.length > self.max_population:
            self._remove_worst_genome()

        if evaluated and unevaluated:
            return self._select_random_genome()
        elif evaluated:
            return self._select_random_evaluated_genome()
        elif unevaluated:
            return self._select_random_unevaluated_genome()
        else:
            raise ValueError(
                "At least one of evaluated or unevaluated must be True")

    def move_genome_to_evaluated(self, genome):
        self._genomes_unevaluated.remove(genome)
        self._genomes_evaluated.append(genome)

    # TODO: Test because it could be wrong
    def get_best_solution(self) -> Genome:
        return max(self._genomes_evaluated, key=lambda genome: genome.fitness)

    # TODO: Test because it could be wrong
    def _remove_worst_genome(self):
        worst_genome = min(self._genomes_evaluated,
                           key=lambda genome: genome.fitness)
        self._genomes_evaluated.remove(worst_genome)
        self.length -= 1

    def _create_genome(self):
        new_genome = self._build_genome()
        self._genomes_unevaluated.append(new_genome)
        self.length += 1

    def _select_random_unevaluated_genome(self) -> Genome:
        selected_genome = random.choice(self._genomes_unevaluated)
        return selected_genome

    def _select_random_evaluated_genome(self) -> Genome:
        selected_genome = random.choice(self._genomes_unevaluated)
        return selected_genome

    def _select_random_genome(self) -> Genome:
        selected_genome = random.choice(
            self._genomes_unevaluated + self._genomes_evaluated)
        return selected_genome

    def _build_genome(self) -> Genome:
        selected_parent = self._select_genome_tournament(
            math.ceil(self.length*0.1))
        child = selected_parent.copy()
        child.mutate()
        return child

    def _select_genome_tournament(self, num_candidates) -> Genome:
        candidates = random.sample(self._genomes_evaluated, num_candidates)
        fittest_genome = max(candidates, key=lambda genome: genome.fitness)
        return fittest_genome
