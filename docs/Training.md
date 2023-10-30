# How to train the model

___

## Overwrite fitness function

    def fitness_function(self):
        fitness = 0
        for neuron in self.output_neurons:
            fitness -= abs(neuron.value - neuron.expected_value)
        return fitness

    NeuralNetwork.get_fitness = fitness_function

In the code example above, a function is created that determines the fitness of a neural network.
The fitness function is called by the `get_fitness` method of the `NeuralNetwork` class.
The important thing is the last line, where we overwrite the default get_fitness method with our new method.
The `self` parameter is important. It is a reference to the object itself. In this case, it is a reference to the neural
network object.

## 