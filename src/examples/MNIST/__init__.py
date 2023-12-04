import numpy as np
import pandas as pd

from src.neural_network import Genome
from src.neural_network.NeuroEvolution import NeuroEvolution


def main():
    def read_mnist(file_path):
        data = pd.read_csv(file_path)
        data = data.iloc[1:].values
        return data

    # Data begins at the second row
    # First column is the label
    # The rest are the pixels
    # Get MNIST data here: https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
    mnist_data = read_mnist('mnist_train.csv')
    length = mnist_data.shape[0]

    yane = NeuroEvolution()
    yane.set_number_of_outputs(10)

    yane.set_max_generations(100)
    yane.set_min_fitness(length)

    def evaluate(genome: Genome):
        fitness = 0

        for row in mnist_data:
            label = row[0]
            pixels = row[1:]

            outputs = genome.forward_propagation(pixels, True)
            result = np.argmax(outputs)

            if result == label:
                fitness += 1
                
        return fitness

    yane.train(evaluate)

    yane.print()
    best_genome = yane.get_best_species_genome()[1]

    print()
    print(best_genome.print())
    print()

    fitness = 0

    print("Testing on training data...")
    for row in mnist_data:
        label = row[1]
        pixels = row[2:]

        outputs = best_genome.forward_propagation(pixels, True)
        result = np.argmax(outputs)

        if result == label:
            fitness += 1
            print("Fitness: " + str(fitness), end="\r")

        print("Output: " + str(result) + " Expected: " + str(label))

    mnist_data = read_mnist('mnist_test.csv')

    fitness = 0

    print("Testing on test data...")
    for row in mnist_data:
        label = row[1]
        pixels = row[2:]

        outputs = best_genome.forward_propagation(pixels, True)
        result = np.argmax(outputs)

        if result == label:
            fitness += 1
            print("Fitness: " + str(fitness), end="\r")

        print("Output: " + str(result) + " Expected: " + str(label))

    print("Got " + str(fitness) + " out of " + str(length) + " correct.")
    best_genome.plot()


if __name__ == "__main__":
    main()
