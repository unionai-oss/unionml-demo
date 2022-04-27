import torch
import torch.nn
from typing import List
from flytekit import workflow, reference_task, dynamic

from pictionary_app.dataset import QuickDrawDataset
from pictionary_app.main import model


@reference_task(
    project="unionml",
    domain="development",
    name="quickdraw_classifier.predict_from_features_task",
    version="a5a044fa67b407fcf84afd0185f1c4b88c99144b",
)
def predict_from_features_task(model_object: torch.nn.Module, features: QuickDrawDataset) -> torch.Tensor:
    ...


@dynamic
def run_batch_predictions(model_object: torch.nn.Module, feature_list: List[torch.Tensor]) -> List[dict]:
    predictions = []
    for features in feature_list:
        predictions.append(predict_from_features_task(model_object=model_object, features=features))
    return predictions


@workflow
def wf(
    model_object: torch.nn.Module,
    feature_list: List[torch.Tensor],
) -> List[dict]:
    return run_batch_predictions(model_object=model_object, feature_list=feature_list)
