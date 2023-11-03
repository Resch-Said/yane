import json


def get_input_data(dataset, pos):
    return dataset[pos]['input']


def get_output_data(dataset, pos):
    return dataset[pos]['output']


def get_data_size(dataset):
    return len(dataset)


def load_data(file_path='dataset.json'):
    with open(file_path) as dataset_file:
        return json.load(dataset_file)
