import torch
import torch.nn
from typing import List
from flytekit import task, workflow, reference_task, dynamic

from pictionary_app.dataset import QuickDrawDataset
from pictionary_app.main import model

@task
def load_quickdraw_data(max_items_per_class: int, class_limit: int) -> QuickDrawDataset:
    return QuickDrawDataset(root="./tmp", max_items_per_class=max_items_per_class, class_limit=class_limit)

@reference_task(
    project="unionml",
    domain="development",
    name="quickdraw_classifier.predict_from_features_task",
    version="TODO",
)
def predict_from_features_task(features: QuickDrawDataset, model: torch.nn.Module) -> torch.Tensor:
    ...

@dynamic
def run_batch_predictions(data: QuickDrawDataset, model: torch.nn.Module) -> List[torch.Tensor]:
    predictions = []
    for features in data:
        predictions.append(predict_from_features_task(features=features, model=model)

    return predictions


@workflow
def wf(max_items_per_class: int, class_limit: int) -> (QuickDrawDataset, List[torch.Tensor]:
    data = load_quickdraw_data(max_items_per_class=max_items_per_class, class_limit=class_limit)
    predictions = run_batch_predictions(data=data, model=model.artifact)
    return data, predictions
