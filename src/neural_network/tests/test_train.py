from src.neural_network import TrainingData

training_data = TrainingData.load_data()

# def test_train():
#    def custom_evaluation(self):
#        fitness = 0
#
#        for j in range(TrainingData.get_data_size(training_data)):
#            input_data = TrainingData.get_input_data(training_data, j)
#            expected_output = TrainingData.get_output_data(training_data, j)
#
#            while len(expected_output) != len(self.output_neurons):
#                new_output_neuron = OutputNeuron()
#                self.add_output_neuron(new_output_neuron)
#
#            output_data = self.forward_propagation(input_data)
#
#            for i, v in enumerate(output_data):
#                fitness -= np.abs(expected_output[i] - v)
#
#        return fitness
#
#    NeuralNetwork.custom_evaluation = custom_evaluation
#
#    yane = NeuroEvolution()
#
#    yane.train(min_fitness=-0.1)
#    yane.print()
