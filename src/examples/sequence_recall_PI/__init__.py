from src.examples import TrainingData

dataset = TrainingData.load_data('dataset_PI.json')

for i in range(10):
    print(f"input: {dataset[i]['input']}")
    print(f"output: {dataset[i]['output']}")
    print()
