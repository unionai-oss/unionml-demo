import torch
import torch.nn
from typing import List, NamedTuple
from flytekit import workflow, map_task, reference_task, task

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


class MapItem(NamedTuple):
   model_object: torch.nn.Module
   features: torch.Tensor


@task
def prepare_map_inputs(model_object: torch.nn.Module, feature_list: List[torch.Tensor]) -> List[MapItem]:
    return [MapItem(model_object, features) for features in feature_list]


@task
def mappable_task(input: MapItem) -> dict:
    return predict_from_features_task(model_object=input.model_object, features=input.features)


@task
def run_batch_predictions(map_input: MapItem) -> List[dict]:
    return map_task(mappable_task)(input=map_input)


@workflow
def wf(
    model_object: torch.nn.Module,
    feature_list: List[torch.Tensor],
) -> List[dict]:
    map_input = prepare_map_inputs(model_object=model_object, feature_list=feature_list)
    return run_batch_predictions(map_input=map_input)
