from src.examples import TrainingData

dataset = TrainingData.load_data('dataset_XOR.json')

length = len(dataset)

for i in range(length):
    print(f"input: {dataset[i]['input']}")
    print(f"output: {dataset[i]['output']}")
    print()
