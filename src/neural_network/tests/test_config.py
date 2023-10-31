from src.neural_network.YaneConfig import *


def test_create_default_json():
    create_default_json_config()


def test_load_default_config_from_json():
    assert load_json_config()["allow_early_output"] is True
    assert load_json_config()["clear_on_new_input"] is True
    assert load_json_config()["fire_rate"] == [1, 10]
    assert load_json_config()["fire_rate_shift"] == [1, 1]
    assert load_json_config()["weight_shift"] == [0.01, 0.1]
    assert load_json_config()["activation_functions"] == ["Tanh", "ReLU", "Sigmoid", "Binary", "Linear"]
    assert load_json_config()["binary_threshold"] == 0.5
    assert load_json_config()["mutation_connection"] == 0.1
    assert load_json_config()["mutation_neuron"] == 0.1
    assert load_json_config()["mutation_weight"] == 0.1
    assert load_json_config()["mutation_fire_rate"] == 0.1
    assert load_json_config()["mutation_activation_function"] == 0.1


def test_load_default_config_from_json_2():
    assert get_allow_early_output() is True
    assert get_clear_on_new_input() is True
    assert get_fire_rate_max() == 10
    assert get_fire_rate_min() == 1
    assert get_random_fire_rate_shift() in range(get_fire_rate_min(), get_fire_rate_max() + 1)
    assert get_random_weight_shift()  # Hard to test
    assert get_random_activation_function() in ["Tanh", "ReLU", "Sigmoid", "Binary", "Linear"]
    assert get_binary_threshold() == 0.5
    assert get_mutation_connection() == 0.1
    assert get_mutation_neuron() == 0.1
    assert get_mutation_weight() == 0.1
    assert get_mutation_fire_rate_probability() == 0.1
    assert get_mutation_activation_function() == 0.1
