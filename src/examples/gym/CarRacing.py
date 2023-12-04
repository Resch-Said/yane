import gymnasium as gym
import numpy as np
from PIL import Image

from src.neural_network.Genome import Genome
from src.neural_network.NeuroEvolution import NeuroEvolution


# needs to be installed
# pip install gym
# pip install gym[box2d]
# pip install swig


def main():
    env = gym.make("CarRacing-v2")

    yane = NeuroEvolution()
    yane.set_number_of_outputs(3)
    yane.set_min_fitness(100)

    def evaluate(genome: Genome):
        state = env.reset()
        state = state[0]
        done = False
        fitness = 0
        for _ in range(100):
            state_image = Image.fromarray(state)
            state_image = state_image.resize((8, 8))
            state_image = state_image.convert('L')

            input_data = np.array(state_image).flatten()
            # input_data = state.flatten()

            outputs = genome.tick(input_data)

            state, reward, done, _, _ = env.step(outputs)
            fitness += reward

        return fitness

    yane.train(evaluate)

    yane.print()
    best_genome = yane.get_best_species_genome()[1]

    print()
    print(best_genome.print())
    print()

    print("Testing best genome")

    env = gym.make('CarRacing-v2', render_mode="human")

    state = env.reset()
    state = state[0]
    done = False
    fitness = 0
    while not done:
        state_image = Image.fromarray(state)
        state_image = state_image.resize((8, 8))
        state_image = state_image.convert('L')

        input_data = np.array(state_image).flatten()
        # input_data = state.flatten()

        outputs = best_genome.tick(input_data)

        state, reward, done, _, _ = env.step(outputs)
        fitness += reward
        print("Fitness: " + str(fitness), end="\r")

        if fitness < 0:
            done = True

    best_genome.plot()

    env.close()


if __name__ == "__main__":
    main()
