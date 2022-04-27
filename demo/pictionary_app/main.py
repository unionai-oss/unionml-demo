import os

from datetime import datetime
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import TrainingArguments, EvalPrediction
    
from ulearn import Dataset, Model

from .dataset import QuickDrawDataset, get_quickdraw_class_names, quickdraw_collate_fn
from .trainer import QuickDrawTrainer, init_model, quickdraw_compute_metrics


dataset = Dataset(name="quickdraw_dataset", test_size=0.2, shuffle=True)
model = Model(name="quickdraw_classifier", init=init_model, dataset=dataset)

# attach Flyte remote backend
model.remote(
    registry="ghcr.io/unionai-oss",
    dockerfile="Dockerfile",
    config_file_path="config/config-remote.yaml",
    project="ulearn",
    domain="development",
)


@dataset.reader
def reader(data_dir: str, max_examples_per_class: int = 1000, class_limit: int = 5) -> QuickDrawDataset:
    return QuickDrawDataset(data_dir, max_examples_per_class, class_limit=class_limit)

# TODO: consider a loader function in the case that flyte doesn't know how to effectively deserialize a dataset
# automatically via types

QuickDrawDatasetType = Union[QuickDrawDataset, torch.utils.data.Subset]


@dataset.splitter
def splitter(
    dataset: QuickDrawDatasetType,
    test_size: float,
    shuffle: bool,
    random_state: int,
) -> torch.utils.data.Subset:
    return dataset.split(test_size)


@dataset.parser
def parser(
    dataset: QuickDrawDatasetType,
    features: Optional[List[str]],
    targets: List[str]
) -> Tuple[QuickDrawDatasetType]:
    return (dataset, )


@dataset.feature_processor
def feature_processor(dataset: Tuple[QuickDrawDatasetType]) -> torch.Tensor:
    # TODO: figure out how to auto-unpack this.
    dataset, *_ = dataset
    features = []
    for i in range(len(dataset)):
        X, _ = dataset[i]
        features.append(X)
    features = torch.stack(features)
    return features


@model.trainer
def trainer(
    module: nn.Module,
    dataset: torch.utils.data.Subset,
    *,
    num_epochs: int = 20
) -> nn.Module:
    timestamp = datetime.now().strftime('%Y-%m-%d-%H%M%S')
    training_args = TrainingArguments(
        output_dir=f'./.tmp/outputs_20k_{timestamp}',
        save_strategy='epoch',
        report_to=['tensorboard'],  # Update to just tensorboard if not using wandb
        logging_strategy='steps',
        logging_steps=100,
        per_device_train_batch_size=256,
        per_device_eval_batch_size=256,
        learning_rate=0.003,
        fp16=torch.cuda.is_available(),
        dataloader_drop_last=True,
        num_train_epochs=num_epochs,
        run_name=f"quickdraw-med-{timestamp}",  # Can remove if not using wandb
        warmup_steps=10000,
        save_total_limit=5,
    )
    quickdraw_trainer = QuickDrawTrainer(
        module,
        training_args,
        data_collator=quickdraw_collate_fn,
        train_dataset=dataset,
        tokenizer=None,
        compute_metrics=quickdraw_compute_metrics,
    )
    train_results = quickdraw_trainer.train()
    quickdraw_trainer.save_model()
    quickdraw_trainer.log_metrics("train", train_results.metrics)
    quickdraw_trainer.save_metrics("train", train_results.metrics)
    quickdraw_trainer.save_state()
    return module


@model.predictor
def predictor(module: nn.Module, features: torch.Tensor) -> torch.Tensor:
    return module(features)


@model.evaluator
def evaluator(module: nn.Module, dataset: torch.utils.data.Subset) -> float:
    top1_acc = []
    for features, label_ids in torch.utils.data.DataLoader(dataset, batch_size=256):
        predictions = predictor(module, features)
        metrics = quickdraw_compute_metrics(EvalPrediction(predictions, label_ids))
        top1_acc.append(metrics["acc1"])
    return float(sum(top1_acc) / len(top1_acc))


if __name__ == "__main__":
    import gradio as gr

    num_classes = 3
    trained_model, metrics = model.train(
        hyperparameters={"num_classes": num_classes},
        trainer_kwargs={"num_epochs": 1},
        data_dir="./.tmp/data",
        max_examples_per_class=1000,
        class_limit=num_classes,
    )

    predictions = model.predict(data_dir="./.tmp/prediction_data", max_examples_per_class=5, class_limit=3)
    class_names = get_quickdraw_class_names()

    def predict(img):
        x = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.
        with torch.no_grad():
            out = model.artifact.object(x)
        probabilities = nn.functional.softmax(out[0], dim=0)
        values, indices = torch.topk(probabilities, num_classes)
        confidences = {class_names[i]: v.item() for i, v in zip(indices, values)}
        return confidences

    gr.Interface(
        fn=predict,
        inputs="sketchpad",
        outputs="label",
        live=True,
    ).launch(share=True)
