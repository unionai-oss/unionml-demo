import random
import torch
import torch.nn
from typing import List, Tuple
from flytekit import workflow, map_task, reference_task, task

from pictionary_app.dataset import QuickDrawDataset
from pictionary_app.dataset import get_quickdraw_class_names
from pictionary_app.main import model, predictor


class MapItem:
    def __init__(self, model_object: torch.nn.Module, features: torch.Tensor):
        self.model_object = model_object
        self.features = features


@task(cache=True, cache_version="v4")
def prepare_map_inputs(
    model_object: torch.nn.Module, feature_list: List[torch.Tensor]
) -> List[MapItem]:
    return [MapItem(model_object, features) for features in feature_list]


@task(cache=True, cache_version="v5")
def prediction_task(
    model_object: torch.nn.Module,
    feature_list: List[torch.Tensor],
) -> List[dict]:
    predictions = []
    for features in feature_list:
        predictions.append(
            predictor(
                model_object,
                torch.tensor(
                    features, dtype=torch.float32
                ).unsqueeze(0).unsqueeze(0),
            )
        )
    return predictions


@task(cache=True, cache_version="v5")
def mappable_prediction_task(input: MapItem) -> dict:
    return predictor(
        input.model_object,
        torch.tensor(
            input.features, dtype=torch.float32
        ).unsqueeze(0).unsqueeze(0),
    )


@task(cache=True, cache_version="v4")
def download_quickdraw_dataset(
    max_items_per_class: int, num_classes: int
) -> QuickDrawDataset:
    return QuickDrawDataset(
        "./quickdraw_data",
        max_items_per_class=max_items_per_class,
        class_limit=num_classes,
    )


@task(cache=True, cache_version="v4")
def generate_input(
    n_samples: int, dataset: QuickDrawDataset,
) -> Tuple[List[torch.Tensor], List[str]]:
    feature_list = []
    label_list = []
    # get a few random entries from the original dataset
    for i in range(n_samples):
        X, y = dataset[random.randint(0, len(dataset) - 1)]
        feature_list.append(X.squeeze())
        label_list.append(dataset.classes[y])
    return feature_list, label_list
