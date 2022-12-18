from ludwig.api import LudwigModel
import pandas as pd

config = {
    "input_features": [
        {
            "name": "source_code",
            "type": "text",
            "encoder": {
                "type": "bert",
                "max_sequence_length": 3000,
            },
        },
    ],
    "output_features": [
        {
            "name": "test_code",
            "type": "text",
        },
    ],
}

model = LudwigModel(config)
model.train(
    training_set="./dataset_training.csv",
    test_set="./dataset_test.csv",
)
model.save("trained")
stats = model.evaluate(dataset="./dataset_test.csv")
print(stats)
prediction_dataset = pd.read_csv("./dataset_test.csv")
prediction_result, output = model.predict(dataset=prediction_dataset)
results = prediction_dataset.join(prediction_result)
for index, row in results.iterrows():
    print(row)
