import numpy as np

from src.examples import TrainingData
from src.neural_network.NeuroEvolution import NeuroEvolution
from src.neural_network.genome.Genome import Genome

dataset = TrainingData.load_data('dataset_XOR.json')

length = len(dataset)

yane = NeuroEvolution()
yane.set_min_fitness(0)
yane.set_number_of_outputs(1)


def evaluate(genome: Genome):
    fitness = 0.0
    for sample in dataset:
        data_input = sample['input']
        target_output = sample['output']

        genome.forward_propagation(data_input)
        predicted_output = genome.get_output_data()

        fitness -= np.abs(predicted_output[0] - target_output[0])

    return fitness


yane.train(evaluate)

yane.print()
best_genome = yane.get_best_species_genome()[1]

print()
print(best_genome.print())
print()

for data in dataset:
    inputs = data['input']
    expected_output = data['output']

    best_genome.forward_propagation(inputs)
    output = best_genome.get_output_data()

    print("Input: " + str(inputs) + " Output: " + str(output) + " Expected: " + str(expected_output))
best_genome.plot()
