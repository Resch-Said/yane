import json
import os
import random


def get_allow_early_output(json_config):
    return json_config["allow_early_output"]


def get_clear_on_new_input(json_config):
    return json_config["clear_on_new_input"]


def get_random_weight_shift(json_config):
    weight_shift = random.uniform(json_config["weight_shift"][0], json_config["weight_shift"][1])
    return weight_shift


def get_random_activation_function(json_config):
    return random.choice(json_config["activation_functions"])


def get_binary_threshold(json_config):
    return json_config["binary_threshold"]


def get_mutation_connection_probability(json_config):
    return json_config["mutation_connection_probability"]


def get_mutation_node_probability(json_config):
    return json_config["mutation_node_probability"]


def get_mutation_weight_probability(json_config):
    return json_config["mutation_weight_probability"]


def get_mutation_activation_function_probability(json_config):
    return json_config["mutation_activation_function_probability"]


def get_random_mutation_weight(json_config):
    return random.uniform(get_mutation_weight_min(json_config), get_mutation_weight_max(json_config))


def get_mutation_weight_min(json_config):
    return json_config["mutation_weight"][0]


def get_mutation_weight_max(json_config):
    return json_config["mutation_weight"][1]


def get_mutation_shift_probability(json_config):
    return json_config["mutation_shift_probability"]


def get_mutation_enabled_probability(json_config):
    return json_config["mutation_enabled_probability"]


def get_species_stagnation_duration(json_config):
    return json_config["species_stagnation_duration"]


def get_species_size_reference(json_config):
    return json_config["species_size_reference"]


def get_max_species_per_population(json_config):
    return json_config["max_species_per_population"]


def get_net_cost_factor(json_config):
    return json_config["net_cost_factor"]


def get_max_population_size(json_config):
    return json_config["max_species_per_population"] * json_config["species_size_reference"]


def get_species_compatibility_node_factor(yane_config):
    return yane_config["species_compatibility_node_factor"]


def get_species_compatibility_connection_factor(yane_config):
    return yane_config["species_compatibility_connection_factor"]


def get_species_compatibility_weight_factor(yane_config):
    return yane_config["species_compatibility_weight_factor"]


def get_reproduction_fraction(json_config):
    return json_config["reproduction_fraction"]


def get_max_bad_reproductions_in_row(json_config):
    return json_config["max_bad_reproductions_in_row"]


def get_improvement_threshold(json_config):
    return json_config["improvement_threshold"]


# TODO: Automatically stop training if no improvement after x generations

def create_default_json_config():
    json_config = {
        # if true, yane can cancel forward propagation if it thinks the output is already good enough. Might increase
        # the training time since an additional output is added, but it can reduce the execution time of forward
        # propagation. If false, yane will always execute forward propagation until the end.
        "allow_early_output": True,
        "clear_on_new_input": True,  # if true, yane will reset all node values when a new input is given
        "weight_shift": [0.01, 0.1],  # How much the weight can change shift in one direction. [min, max]
        "mutation_weight": [-2, 2],  # The range of the random weight when mutating
        "activation_functions": ["Tanh", "ReLU", "Sigmoid", "Binary", "Linear"],  # all activation functions
        "binary_threshold": 0.5,  # only used for binary activation function
        "mutation_connection_probability": 0.1,  # chance that a new connection is created
        "mutation_node_probability": 0.1,  # chance that a new node is created
        "mutation_weight_probability": 0.1,  # chance that a weight is mutated
        "mutation_shift_probability": 0.5,  # chance that a weight is shifted
        "mutation_activation_function_probability": 0.1,  # chance that an activation function is mutated
        "mutation_enabled_probability": 0.1,  # chance that a connection is enabled/disabled
        # The number of generations without improvement until a species is considered stagnant
        "species_stagnation_duration": 5,
        "species_size_reference": 50,  # The approximate amount of genomes in a species. May fluctuate.
        "max_species_per_population": 5,  # The approximate amount of species in a population. May fluctuate.
        "species_compatibility_node_factor": 1,  # The factor that is multiplied with the node difference
        "species_compatibility_connection_factor": 1,  # The factor that is multiplied with the connection difference
        "species_compatibility_weight_factor": 0.4,  # The factor that is multiplied with the weight difference
        "reproduction_fraction": 0.2,  # The fraction of the population that is allowed to reproduce
        # The maximum amount of times a genome is allowed to make bad genomes in a row
        "max_bad_reproductions_in_row": 5,
        "improvement_threshold": 0.01,  # The minimum improvement that is required to consider a species improved

        # The factor that is multiplied with the net cost to calculate the fitness.
        # If to high, the net cost will be prioritized over the fitness.
        "net_cost_factor": 0.0001
    }
    with open('yane_config.json', 'w') as json_config_file:
        json.dump(json_config, json_config_file)


def load_json_config():
    if not os.path.exists('yane_config.json'):
        create_default_json_config()
    with open('yane_config.json') as json_config_file:
        json_config = json.load(json_config_file)
    return json_config
