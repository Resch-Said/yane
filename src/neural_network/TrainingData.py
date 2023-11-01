import json


def get_input_data(pos):
    return load_data()[pos]['input']


def get_output_data(pos):
    return load_data()[pos]['output']


def get_data_size():
    return len(load_data())


def load_data(file_path='dataset.json'):
    with open(file_path) as dataset_file:
        return json.load(dataset_file)
