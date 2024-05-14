import unittest

from src.neural_network.util.activation import ActivationFunction


class TestActivationFunction(unittest.TestCase):

    def test_get_function(self):
        self.assertEqual(ActivationFunction.get_function(
            'linear'), ActivationFunction.LINEAR)
        self.assertEqual(ActivationFunction.get_function(
            'sigmoid'), ActivationFunction.SIGMOID)
        self.assertEqual(ActivationFunction.get_function(
            'tanh'), ActivationFunction.TANH)
        self.assertEqual(ActivationFunction.get_function(
            'relu'), ActivationFunction.RELU)
        self.assertEqual(ActivationFunction.get_function(
            'binary'), ActivationFunction.BINARY)
        with self.assertRaises(Exception):
            ActivationFunction.get_function('nonexistent')

    def test_activate(self):
        self.assertEqual(ActivationFunction.activate(
            ActivationFunction.LINEAR, 1), 1)
        self.assertEqual(ActivationFunction.activate(
            ActivationFunction.SIGMOID, 0), 0.5)
        self.assertEqual(ActivationFunction.activate(
            ActivationFunction.TANH, 0), 0)
        self.assertEqual(ActivationFunction.activate(
            ActivationFunction.RELU, -1), 0)
        self.assertIn(ActivationFunction.activate(
            ActivationFunction.BINARY, 0), [0, 1])
        with self.assertRaises(Exception):
            ActivationFunction.activate('nonexistent', 0)

    def test_linear(self):
        self.assertEqual(ActivationFunction.linear(1), 1)

    def test_sigmoid(self):
        self.assertEqual(ActivationFunction.sigmoid(0), 0.5)

    def test_tanh(self):
        self.assertEqual(ActivationFunction.tanh(0), 0)

    def test_relu(self):
        self.assertEqual(ActivationFunction.relu(-1), 0)

    def test_binary(self):
        self.assertIn(ActivationFunction.binary(0), [0, 1])


if __name__ == '__main__':
    unittest.main()
