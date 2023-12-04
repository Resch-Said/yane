import gymnasium as gym
import numpy as np
from PIL import Image

from src.neural_network.Genome import Genome
from src.neural_network.NeuroEvolution import NeuroEvolution

# needs to be installed
# pip install gym
# pip install gym[box2d]
# pip install swig


env = gym.make("CarRacing-v2")

yane = NeuroEvolution()
yane.set_number_of_outputs(3)
yane.set_min_fitness(500)


# TODO: To many input nodes, so we should update yane to be able to create the forward order list
#  starting from the outputs
def evaluate_normal_input(genome: Genome):
    state = env.reset()
    state = state[0]
    done = False
    fitness = 0
    while not done:
        state_image = Image.fromarray(state)
        state_image = state_image.resize((2, 2))
        state_image = state_image.convert('L')

        input_data = np.array(state_image).flatten()
        outputs = genome.forward_propagation(input_data)

        state, reward, done, _, _ = env.step(outputs)
        fitness += reward

        if fitness < 0:
            done = True

    return fitness


yane.train(evaluate_normal_input)

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
    input_data = np.convolve(state.flatten(), [0.5, 0.5, 0.5, 0.5, 0.5])

    outputs = best_genome.forward_propagation(input_data)

    state, reward, done, _, _ = env.step(outputs)
    fitness += reward
    print("Fitness: " + str(fitness), end="\r")

best_genome.plot()

env.close()
