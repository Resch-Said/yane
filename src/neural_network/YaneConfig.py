import json
import os
import random


def get_allow_early_output(json_config):
    return json_config["allow_early_output"]


def get_clear_on_new_input(json_config):
    return json_config["clear_on_new_input"]


def get_fire_rate_max(json_config):
    return json_config["fire_rate"][1]


def get_fire_rate_min(json_config):
    return json_config["fire_rate"][0]


def get_random_fire_rate(json_config):
    return random.randint(json_config["fire_rate"][0], json_config["fire_rate"][1])


def get_random_weight_shift(json_config):
    weight_shift = random.uniform(json_config["weight_shift"][0], json_config["weight_shift"][1])
    return weight_shift


def get_random_activation_function(json_config):
    return random.choice(json_config["activation_functions"])


def get_binary_threshold(json_config):
    return json_config["binary_threshold"]


def get_mutation_connection_probability(json_config):
    return json_config["mutation_connection_probability"]


def get_mutation_neuron_probability(json_config):
    return json_config["mutation_neuron_probability"]


def get_mutation_weight_probability(json_config):
    return json_config["mutation_weight_probability"]


def get_mutation_fire_rate_probability(json_config):
    return json_config["mutation_fire_rate_probability"]


def get_mutation_activation_function_probability(json_config):
    return json_config["mutation_activation_function_probability"]


def get_mutation_random_weight(json_config):
    return random.uniform(json_config["mutation_random_weight"][0],
                          json_config["mutation_random_weight"][1])


def get_population_size(json_config):
    return json_config["population_size"]


def get_population_survival_rate(json_config):
    return json_config["population_survival_rate"]


def create_default_json_config():
    json_config = {
        # if true, yane can cancel forward propagation if it thinks the output is already good enough. Might increase
        # the training time since an additional output is added, but it can reduce the execution time of forward
        # propagation. If false, yane will always execute forward propagation until the end.
        "allow_early_output": True,
        "clear_on_new_input": True,  # if true, yane will reset all neuron values when a new input is given
        "fire_rate": [1, 10],  # How often a neuron is allowed to fire before the next input. [min, max]
        "weight_shift": [0.01, 0.1],  # How much the weight can change when optimizing. [min, max]
        "mutation_random_weight": [-1, 1],  # The range of the random weight when mutating
        "activation_functions": ["Tanh", "ReLU", "Sigmoid", "Binary", "Linear"],  # all activation functions
        "binary_threshold": 0.5,  # only used for binary activation function
        "mutation_connection_probability": 0.5,  # chance that a new connection is created
        "mutation_neuron_probability": 0.5,  # chance that a new neuron is created
        "mutation_weight_probability": 0.5,  # chance that a weight is mutated
        "mutation_fire_rate_probability": 0.5,  # chance that a fire rate is mutated
        "mutation_activation_function_probability": 0.5,  # chance that an activation function is mutated
        "population_size": 50,  # the amount of neural networks in the population
        "population_survival_rate": 0.5,  # the amount of neural networks that survive
    }
    with open('yane_config.json', 'w') as json_config_file:
        json.dump(json_config, json_config_file)


def load_json_config():
    if not os.path.exists('yane_config.json'):
        create_default_json_config()
    with open('yane_config.json') as json_config_file:
        json_config = json.load(json_config_file)
    return json_config
