import numpy as np

from src.examples import TrainingData
from src.neural_network.Genome import Genome
from src.neural_network.NeuroEvolution import NeuroEvolution

dataset = TrainingData.load_data('dataset_PI.json')

length = len(dataset)

yane = NeuroEvolution()
yane.set_max_generations(10000)
yane.set_number_of_outputs(10)


def evaluate(genome: Genome):
    fitness = 0.0

    best_fitness = yane.get_best_fitness()
    if best_fitness is None:
        best_fitness = 0.0

    for sample in dataset[:1 + int(np.round(best_fitness))]:
        data_input = sample['input']
        target_output = sample['output']

        genome.forward_propagation(data_input)
        predicted_output = genome.get_outputs()

        highest = np.argmax(predicted_output)

        if highest == target_output[0]:
            fitness += 1.0

    return fitness


yane.train(evaluate)

best_fitness = yane.get_best_fitness()
best_genome = yane.get_genomes_population()[0]

yane.print()
print()
print(best_genome.print())
print()

for data in dataset[:1 + int(np.round(best_fitness))]:
    inputs = data['input']
    expected_output = data['output']

    best_genome.forward_propagation(inputs)
    output = best_genome.get_outputs()

    highest = np.argmax(output)

    print("Input: " + str(inputs) + " Output: " + str(highest) + " Expected: " + str(expected_output))
