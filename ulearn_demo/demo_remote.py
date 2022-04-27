from pictionary_app.main import model

num_classes = 5


model.remote_train(
    hyperparameters={"num_classes": num_classes},
    trainer_kwargs={"num_epochs": 1},
    data_dir="./data",
    max_examples_per_class=2000,
    class_limit=num_classes,
)
