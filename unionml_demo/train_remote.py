from pictionary_app import model

num_classes = 345

# run training on a remote
execution = model.remote_train(
    wait=False,
    hyperparameters={"num_classes": num_classes},
    trainer_kwargs={"num_epochs": 3, "batch_size": 2048},
    data_dir="./data",
    max_examples_per_class=3000,
    class_limit=num_classes,
)
