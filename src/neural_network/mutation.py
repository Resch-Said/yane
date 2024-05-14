from enum import Enum
import random
from re import M
import numpy as np


class MutationType(Enum):
    MUTATION_RATE = "mutation_rate"  # Probability to mutate other mutations
    MUTATION_SHIFT_RATE = "mutation_shift_rate"
    MUTATION_CUSTOM_RATE = "mutation_custom_rate"
    MUTATION_BIAS_RATE = "mutation_bias_rate"
    MUTATION_BOOL_RATE = "mutation_bool_rate"


class Mutation:
    def __init__(self):
        self.mutation_rate = {MutationType.MUTATION_RATE: 0.1,
                              MutationType.MUTATION_SHIFT_RATE: 0.1,
                              MutationType.MUTATION_CUSTOM_RATE: 0.1,
                              MutationType.MUTATION_BIAS_RATE: 0.1,
                              MutationType.MUTATION_BOOL_RATE: 0.1}
        self.value_shift_rate = 1
        self.bias = 0.001  # Bias is used to prevent values from being 0

    def mutate_value(self, value):
        if random.random() < self.mutation_rate[MutationType.MUTATION_SHIFT_RATE]:
            return value * self.value_shift_rate + self.bias
        return value

    def mutate_bool(self, value):
        if random.random() < self.mutation_rate[MutationType.MUTATION_BOOL_RATE]:
            return not value
        return value

    def _mutate_bias(self):
        if random.random() < self.mutation_rate[MutationType.MUTATION_BIAS_RATE]:
            if -0.001 < self.bias < 0.001:
                self.bias = 0.001

            self.bias *= self.value_shift_rate

    def _mutate_shift(self):
        if random.random() < self.mutation_rate[MutationType.MUTATION_SHIFT_RATE]:
            if self.value_shift_rate < 0.001:
                self.value_shift_rate = 0.001

            self.value_shift_rate *= np.random.uniform(0.9, 1.1)

    # TODO: Add weight probabilities which element should be chosen. Keyword: random.choices
    def mutate_custom(self, value, enum):
        if random.random() < self.mutation_rate[MutationType.MUTATION_CUSTOM_RATE]:
            return random.choice(list(enum))
        return value

    def mutate_rates(self):
        if random.random() < self.mutation_rate[MutationType.MUTATION_RATE]:
            self.mutation_rate[MutationType.MUTATION_RATE] *= np.random.uniform(
                0.9, 1.1)
            self.mutation_rate[MutationType.MUTATION_SHIFT_RATE] *= np.random.uniform(
                0.9, 1.1)
            self.mutation_rate[MutationType.MUTATION_CUSTOM_RATE] *= np.random.uniform(
                0.9, 1.1)
            self.mutation_rate[MutationType.MUTATION_BIAS_RATE] *= np.random.uniform(
                0.9, 1.1)
            self.mutation_rate[MutationType.MUTATION_BOOL_RATE] *= np.random.uniform(
                0.9, 1.1)

        self._mutate_bias()
        self._mutate_shift()

        for mutation in self.mutation_rate.values():
            if mutation < 0.001:
                mutation = 0.001
            elif mutation > 0.999:
                mutation = 0.999

    def copy(self):
        mutation = Mutation()
        mutation.mutation_rate = self.mutation_rate.copy()
        mutation.value_shift_rate = self.value_shift_rate
        mutation.bias = self.bias

        return mutation
