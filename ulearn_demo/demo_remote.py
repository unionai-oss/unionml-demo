from pictionary_app.main import model

num_classes = 5


# attach Flyte remote backend
model.remote(
    registry="ghcr.io/unionai-oss",
    dockerfile="Dockerfile",
    config_file_path="config/config-remote.yaml",
    project="unionml",
    domain="development",
)

model.remote_train(
    hyperparameters={"num_classes": num_classes},
    trainer_kwargs={"num_epochs": 1},
    data_dir="./data",
    max_examples_per_class=2000,
    class_limit=num_classes,
)
