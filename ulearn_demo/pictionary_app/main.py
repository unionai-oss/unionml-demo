from typing import Union

import numpy as np
import torch
import torch.nn as nn

from transformers import  EvalPrediction
    
from unionml import Dataset, Model

from .dataset import QuickDrawDataset, get_quickdraw_class_names
from .trainer import init_model, quickdraw_compute_metrics, quickdraw_trainer

QuickDrawDatasetType = Union[QuickDrawDataset, torch.utils.data.Subset]


dataset = Dataset(name="quickdraw_dataset", test_size=0.2, shuffle=True)
model = Model(name="quickdraw_classifier", init=init_model, dataset=dataset)

# attach remote backend to the model
model.remote(
    registry="ghcr.io/unionai-oss",
    dockerfile="Dockerfile",
    config_file_path="config/config-remote.yaml",
    project="unionml",
    domain="development",
)


@dataset.reader(cache=True, cache_version="1")
def reader(data_dir: str, max_examples_per_class: int = 1000, class_limit: int = 5) -> QuickDrawDataset:
    return QuickDrawDataset(data_dir, max_examples_per_class, class_limit=class_limit)


@dataset.feature_loader
def feature_loader(data: Union[QuickDrawDatasetType, np.ndarray]) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.
    return torch.stack([data[i][0] for i in range(len(data))])


@model.trainer(cache=True, cache_version="1")
def trainer(module: nn.Module, dataset: torch.utils.data.Subset, *, num_epochs: int = 20) -> nn.Module:
    return quickdraw_trainer(module, dataset, num_epochs)


@model.evaluator
def evaluator(module: nn.Module, dataset: torch.utils.data.Subset) -> float:
    top1_acc = []
    for features, label_ids in torch.utils.data.DataLoader(dataset, batch_size=256):
        top1_acc.append(quickdraw_compute_metrics(EvalPrediction(module(features), label_ids))["acc1"])
    return float(sum(top1_acc) / len(top1_acc))


@model.predictor(cache=True, cache_version="1")
def predictor(module: nn.Module, features: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        probabilities = nn.functional.softmax(module(features)[0], dim=0)
    class_names = get_quickdraw_class_names()
    values, indices = torch.topk(probabilities, 3)
    return {class_names[i]: v.item() for i, v in zip(indices, values)}


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

    gr.Interface(
        fn=model.predict,
        inputs="sketchpad",
        outputs="label",
        live=True,
        allow_flagging="never",
    ).launch(share=True)
