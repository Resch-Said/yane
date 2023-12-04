import numpy as np

from src.examples import TrainingData
from src.neural_network.Genome import Genome
from src.neural_network.NeuroEvolution import NeuroEvolution

dataset = TrainingData.load_data('dataset_PI.json')

length = len(dataset)

yane = NeuroEvolution()
yane.set_min_fitness(-0.5)
yane.set_number_of_outputs(10)


# if target_output is 1: the predicted output on the index of the target output should be 1 and the rest should be 0
def calculate_fitness(target_output, predicted_output):
    target_output_index = target_output[0]

    output_value_for_wrong_prediction = 0.4
    output_value_for_correct_prediction = 0.6

    fitness = 0.0

    for i in range(len(predicted_output)):
        if i == target_output_index:
            fitness += 1 - np.abs(
                predicted_output[i] - output_value_for_correct_prediction) / output_value_for_correct_prediction
        else:
            fitness += 1 - np.abs(
                predicted_output[i] - output_value_for_wrong_prediction) / output_value_for_wrong_prediction

    return fitness


def calculate_fitness_2(target_output, predicted_output):
    target_output_index = target_output[0]

    target_output_value_for_wrong_prediction = 0
    target_output_value_for_correct_prediction = 1

    fitness = 0.0

    for i in range(len(predicted_output)):
        if i == target_output_index:
            fitness -= np.abs(predicted_output[i] - target_output_value_for_correct_prediction)
        else:
            fitness -= np.abs(predicted_output[i] - target_output_value_for_wrong_prediction)

    return fitness


decimal_places = 5


def evaluate(genome: Genome):
    fitness = 0.0

    for sample in dataset[:decimal_places]:
        data_input = sample['input']
        target_output = sample['output']

        genome.forward_propagation(data_input)
        predicted_output = genome.get_outputs()

        fitness += calculate_fitness_2(target_output, predicted_output)

    return fitness


yane.train(evaluate)

best_fitness = yane.get_best_fitness()
best_genome = yane.get_best_species_genome()[1]

yane.print()
print()
print(best_genome.print())
print()

for data in dataset[:decimal_places]:
    inputs = data['input']
    expected_output = data['output']

    best_genome.forward_propagation(inputs)
    output = best_genome.get_outputs()

    highest = np.argmax(output)

    print("Input: " + str(inputs) + " Output: " + str(highest) + " Expected: " + str(expected_output))
