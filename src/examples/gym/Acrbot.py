import gym
import numpy as np

from src.neural_network.NeuroEvolution import NeuroEvolution
from src.neural_network.genome.Genome import Genome

env = gym.make('Acrobot-v1')

yane = NeuroEvolution()
yane.set_number_of_outputs(env.action_space.n)
yane.set_min_fitness(-64)


def evaluate(genome: Genome):
    state = env.reset()
    state = state[0]
    done = False
    fitness = 0
    while not done:
        input_data = list(state)

        outputs = genome.tick(input_data)
        action = np.argmax(outputs)

        state, reward, done, _, _ = env.step(action)
        fitness += reward
        if fitness < -100:
            done = True

    return fitness


yane.train(evaluate)

yane.print()
best_genome = yane.get_best_species_genome()[1]

print()
print(best_genome.print())
print()

print("Testing best genome")

env = gym.make('Acrobot-v1', render_mode="human")

state = env.reset()
state = state[0]
done = False
fitness = 0
while not done:
    input_data = list(state)

    outputs = best_genome.tick(input_data)
    action = np.argmax(outputs)

    state, reward, done, _, _ = env.step(action)
    fitness += reward
    print("Fitness: " + str(fitness), end="\r")

best_genome.plot()

env.close()
