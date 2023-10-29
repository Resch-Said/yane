import json
import os
import random


def get_allow_early_output():
    return load_json_config()["allow_early_output"]


def get_clear_on_new_input():
    return load_json_config()["clear_on_new_input"]


def get_fire_rate_max():
    return load_json_config()["fire_rate"][1]


def get_fire_rate_min():
    return load_json_config()["fire_rate"][0]


def get_random_fire_rate_shift():
    return random.randint(load_json_config()["fire_rate_shift"][0], load_json_config()["fire_rate_shift"][1])


def get_random_weight_shift():
    return random.uniform(load_json_config()["weight_shift"][0], load_json_config()["weight_shift"][1])


def get_random_activation_function():
    return random.choice(load_json_config()["activation_functions"])


def get_binary_threshold():
    return load_json_config()["binary_threshold"]


def get_mutation_connection():
    return load_json_config()["mutation_connection"]


def get_mutation_neuron():
    return load_json_config()["mutation_neuron"]


def get_mutation_weight():
    return load_json_config()["mutation_weight"]


def get_mutation_fire_rate():
    return load_json_config()["mutation_fire_rate"]


def get_mutation_activation_function():
    return load_json_config()["mutation_activation_function"]


def create_default_json_config():
    json_config = {
        # if true, yane can cancel forward propagation if it thinks the output is already good enough. Might increase
        # the training time since an additional output is added, but it can reduce the execution time of forward
        # propagation. If false, yane will always execute forward propagation until the end.
        "allow_early_output": True,
        "clear_on_new_input": False,  # if true, yane will reset all neuron values when a new input is given
        "fire_rate": [1, 10],  # How often a neuron is allowed to fire before the next input. [min, max]
        "fire_rate_shift": [1, 1],  # How much the fire rate can change when mutating. [min, max]
        "weight_shift": [0.01, 0.1],  # How much the weight can change when mutating. [min, max]
        "activation_functions": ["Tanh", "ReLU", "Sigmoid", "Binary", "Linear"],  # all activation functions
        "binary_threshold": 0.5,  # only used for binary activation function
        "mutation_connection": 0.1,  # chance that a new connection is created
        "mutation_neuron": 0.1,  # chance that a new neuron is created
        "mutation_weight": 0.1,  # chance that a weight is mutated
        "mutation_fire_rate": 0.1,  # chance that a fire rate is mutated
        "mutation_activation_function": 0.1  # chance that an activation function is mutated
    }
    with open('default_config.json', 'w') as json_config_file:
        json.dump(json_config, json_config_file)


def load_json_config():
    if os.path.exists('config.json'):
        with open('config.json') as json_config_file:
            json_config = json.load(json_config_file)
    else:
        with open('default_config.json') as json_config_file:
            json_config = json.load(json_config_file)
    return json_config
