import gym
import numpy as np

from neural_network.yane import Yane


def main():
    env = gym.make('CartPole-v1')

    yane = Yane()
    # yane.set_number_of_outputs(env.action_space.n)
    yane.set_min_fitness(500)

    def fitness():
        state = env.reset()
        state = state[0]
        done = False
        fitness = 0

        while not done:
            input_data = list(state)

            outputs = yane.step(input_data)
            action = np.argmax(outputs)

            state, reward, done, _, _ = env.step(action)
            fitness += reward
            print("Fitness: " + str(fitness), end="\r")

        return fitness

    yane.run(fitness)

    yane.print()
    best_solution = yane.get_best_solution()

    print()
    print(best_solution.print())
    print()

    env.close()


if __name__ == '__main__':
    main()
