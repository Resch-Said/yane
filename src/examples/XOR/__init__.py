import numpy as np

from src.examples import TrainingData
from src.neural_network.Genome import Genome
from src.neural_network.NeuroEvolution import NeuroEvolution

dataset = TrainingData.load_data('dataset_XOR.json')

length = len(dataset)

yane = NeuroEvolution()
yane.set_max_generations(1000)
yane.set_number_of_outputs(1)


def evaluate(genome: Genome):
    fitness = 0.0
    for sample in dataset:
        data_input = sample['input']
        target_output = sample['output']

        genome.forward_propagation(data_input)
        predicted_output = genome.get_outputs()

        fitness -= np.abs(predicted_output[0] - target_output[0])

    return fitness


yane.train(evaluate)

yane.print()
print()
print(yane.get_genomes_population()[0].print())
print()

for data in dataset:
    inputs = data['input']
    expected_output = data['output']

    yane.get_genomes_population()[0].forward_propagation(inputs)
    output = yane.get_genomes_population()[0].get_outputs()

    print("Input: " + str(inputs) + " Output: " + str(output) + " Expected: " + str(expected_output))
