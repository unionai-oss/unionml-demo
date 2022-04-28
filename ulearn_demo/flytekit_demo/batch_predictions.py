import torch
import torch.nn
from typing import List, NamedTuple
from flytekit import workflow, map_task, reference_task, task

from pictionary_app.dataset import QuickDrawDataset
from pictionary_app.main import model, predictor


class MapItem:
   def __init__(self, model_object: torch.nn.Module, features: torch.Tensor):
       self.model_object = model_object
       self.features = features


@task
def prepare_map_inputs(model_object: torch.nn.Module, feature_list: List[torch.Tensor]) -> List[MapItem]:
    return [MapItem(model_object, features) for features in feature_list]


@task
def mappable_task(input: MapItem) -> dict:
    return predictor(input.model_object, input.features)


@task
def run_batch_predictions(map_input: List[MapItem]) -> List[dict]:
    return map_task(mappable_task)(input=map_input)


@workflow
def wf(
    model_object: torch.nn.Module,
    feature_list: List[torch.Tensor],
) -> List[dict]:
    map_input = prepare_map_inputs(model_object=model_object, feature_list=feature_list)
    return run_batch_predictions(map_input=map_input)
