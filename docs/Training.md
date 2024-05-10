# How to train the model

___

## Overwrite fitness function

    def custom_fitness(self):
        fitness = 0
        for neuron in self.output_neurons:
            fitness -= abs(neuron.value - neuron.expected_value)
        return fitness

    NeuralNetwork.custom_fitness = custom_fitness

In the code example above, a function is created that determines the fitness of a neural network.
You can change the custom fitness however you want. The only requirement is that it returns a fitness and the self
parameter.

The custom fitness function is called by the `get_fitness` method of the `NeuralNetwork` class.
The important thing is the last line, where we overwrite the default `custom_fitness` method with our new method.

##    