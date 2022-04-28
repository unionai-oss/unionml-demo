import random
import torch
import torch.nn
from typing import List
from flytekit import workflow, map_task, reference_task, task

from pictionary_app.dataset import QuickDrawDataset
from pictionary_app.dataset import get_quickdraw_class_names
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
    features = torch.tensor(input.features, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.
    return predictor(input.model_object, features)


batch_predictions_task = map_task(mappable_task)


@task(cache=True, cache_version="v1")
def download_quickdraw_class_names() -> List[str]:
   return get_quickdraw_class_names()


@task(cache=True, cache_version="v1")
def download_quickdraw_dataset(max_items_per_class: int, num_classes: int) -> QuickDrawDataset:
   return QuickDrawDataset("/tmp/quickdraw_data", max_items_per_class=max_items_per_class, class_limit=num_classes)


@task
def generate_input(n_entries: int, dataset: QuickDrawDataset, class_names: List[str], max_items_per_class: int, num_classes: int) -> (List[torch.Tensor], List[str]):
   feature_list = []
   label_list = []
   # Grab a few random entries from the original dataset
   for i in range(n_entries):
      X, y = dataset[random.randint(0, len(dataset) - 1)]
      feature_list.append(X.squeeze())
      label_list.append(class_names[y])
   return feature_list, label_list
