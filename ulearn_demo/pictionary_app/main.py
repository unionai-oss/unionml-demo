from typing import Union

import numpy as np
import torch
import torch.nn as nn

from transformers import  EvalPrediction

from flytekit import Resources    
from unionml import Dataset, Model

from .dataset import QuickDrawDataset, get_quickdraw_class_names
from .trainer import init_model, quickdraw_compute_metrics, quickdraw_trainer


dataset = Dataset(name="quickdraw_dataset", test_size=0.2, shuffle=True)
model = Model(name="quickdraw_classifier", init=init_model, dataset=dataset)

# attach remote backend to the model
model.remote(
    registry="ghcr.io/unionai-oss",
    dockerfile="Dockerfile.gpu",
    config_file_path="config/config-remote.yaml",
    project="unionml",
    domain="development",
)

# define compute resource requirements
reader_resources = Resources(cpu="1", mem="6Gi")
trainer_resources = Resources(gpu="1", mem="6Gi")


@dataset.reader(cache=True, cache_version="1.0", requests=reader_resources, limits=reader_resources)
def reader(data_dir: str, max_examples_per_class: int = 1000, class_limit: int = 5) -> QuickDrawDataset:
    return QuickDrawDataset(data_dir, max_examples_per_class, class_limit=class_limit)


@dataset.feature_loader
def feature_loader(data: Union[QuickDrawDataset, np.ndarray]) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.
    return torch.stack([data[i][0] for i in range(len(data))])


@model.trainer(cache=True, cache_version="1.0", requests=trainer_resources, limits=trainer_resources)
def trainer(module: nn.Module, dataset: torch.utils.data.Subset, *, num_epochs: int = 20) -> nn.Module:
    return quickdraw_trainer(module, dataset, num_epochs)


@model.evaluator
def evaluator(module: nn.Module, dataset: QuickDrawDataset) -> float:
    top1_acc = []
    for features, label_ids in torch.utils.data.DataLoader(dataset, batch_size=256):
        top1_acc.append(quickdraw_compute_metrics(EvalPrediction(module(features), label_ids))["acc1"])
    return float(sum(top1_acc) / len(top1_acc))


@model.predictor(cache=True, cache_version="1.0")
def predictor(module: nn.Module, features: torch.Tensor) -> dict:
    with torch.no_grad():
        probabilities = nn.functional.softmax(module(features)[0], dim=0)
    class_names = get_quickdraw_class_names()
    values, indices = torch.topk(probabilities, 3)
    return {class_names[i]: v.item() for i, v in zip(indices, values)}
