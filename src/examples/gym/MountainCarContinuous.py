import gym

from src.neural_network.NeuroEvolution import NeuroEvolution
from src.neural_network.genome.Genome import Genome

env = gym.make('MountainCarContinuous-v0')

yane = NeuroEvolution()
yane.set_number_of_outputs(1)
yane.set_min_fitness(1000)


def evaluate(genome: Genome):
    state = env.reset()
    state = state[0]
    done = False
    fitness = 0
    for _ in range(800):
        input_data = list(state)

        outputs = genome.tick(input_data)
        action = outputs

        state, reward, done, _, _ = env.step(action)
        fitness += reward + state[1]

    return fitness


yane.train(evaluate)

yane.print()
best_genome = yane.get_best_species_genome()[1]

print()
print(best_genome.print())
print()

print("Testing best genome")

env = gym.make('MountainCarContinuous-v0', render_mode="human")

state = env.reset()
state = state[0]
done = False
fitness = 0
for _ in range(800):
    input_data = list(state)

    outputs = best_genome.tick(input_data)
    action = outputs

    state, reward, done, _, _ = env.step(action)
    fitness += reward + state[1]
    print("Fitness: " + str(fitness), end="\r")

best_genome.plot()

env.close()
