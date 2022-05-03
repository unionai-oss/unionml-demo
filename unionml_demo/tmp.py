from pictionary_app import model

num_classes = 345

# run training on a remote
execution = model.remote_train(
    hyperparameters={"num_classes": num_classes},
    trainer_kwargs={"num_epochs": 1, "batch_size": 4096},
    data_dir="./data",
    max_examples_per_class=10000,
    class_limit=num_classes,
)
