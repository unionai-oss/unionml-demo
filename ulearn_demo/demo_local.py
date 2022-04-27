from pictionary_app.main import model


num_classes = 3
trained_model, metrics = model.train(
    hyperparameters={"num_classes": num_classes},
    trainer_kwargs={"num_epochs": 1},
    data_dir="./.tmp/data",
    max_examples_per_class=1000,
    class_limit=num_classes,
)

predictions = model.predict(data_dir="./.tmp/prediction_data", max_examples_per_class=5, class_limit=3)
print(predictions)
