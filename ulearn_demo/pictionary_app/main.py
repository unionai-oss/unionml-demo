# %%
from typing import Union

import numpy as np
import torch
import torch.nn as nn

from transformers import  EvalPrediction

from flytekit import Resources    
from unionml import Dataset, Model

from pictionary_app.dataset import QuickDrawDataset, get_quickdraw_class_names
from pictionary_app.trainer import init_model, quickdraw_compute_metrics, quickdraw_trainer


# %% [markdown]
# ## Mechnaincs of the UnionML app
# the UnionML App consists of 2 major components,
# 1. **Model**: Model is essential the enclosure that contains the training program and the resultant trained model. Model can be anything that can be run using the ``train`` method and can be saved and re-hydrated.
# 2. **Dataset** Dataset is the actual data that will be used to train. Dataset can be image data or structured data. For fun, we will use the doodle dataset here

dataset = Dataset(name="quickdraw_dataset", test_size=0.2, shuffle=True)
model = Model(name="quickdraw_classifier", init=init_model, dataset=dataset)

# %% [markdown]
# define compute resource requirements

reader_resources = Resources(cpu="1", mem="6Gi")
trainer_resources = Resources(cpu="1", mem="6Gi")

# %% [markdown]
# ### Dataset Reader
#
# This method reads data into the trainer running on Flyte and also defines how to split it into training and test sets.

# %%
@dataset.reader(cache=True, cache_version="1.0", requests=reader_resources, limits=reader_resources)
def reader(data_dir: str, max_examples_per_class: int = 1000, class_limit: int = 5) -> QuickDrawDataset:
    return QuickDrawDataset(data_dir, max_examples_per_class, class_limit=class_limit)


# %% [markdown]
# ### optionally define how features are parsed
# Defines how to parse out features from the dataset we produced above.

# %%
@dataset.feature_loader
def feature_loader(data: Union[QuickDrawDataset, np.ndarray]) -> torch.Tensor:
    if isinstance(data, np.ndarray):
        return torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.
    return torch.stack([data[i][0] for i in range(len(data))])


# %% [markdown]
# ### Train the Model
# Specify how to train your model on Flyte.

# %%
@model.trainer(cache=True, cache_version="1.0", requests=trainer_resources, limits=trainer_resources)
def trainer(module: nn.Module, dataset: torch.utils.data.Subset, *, num_epochs: int = 20) -> nn.Module:
    return quickdraw_trainer(module, dataset, num_epochs)


# %% [markdown]
# ### Evaluation criteria
# To train a model correctly, provide the evaluation criteria. This may be invoked per epoch

# %%
@model.evaluator
def evaluator(module: nn.Module, dataset: QuickDrawDataset) -> float:
    top1_acc = []
    for features, label_ids in torch.utils.data.DataLoader(dataset, batch_size=256):
        if torch.cuda.is_available():
            features = features.to("cuda")
            label_ids = label_ids.to("cuda")
        top1_acc.append(quickdraw_compute_metrics(EvalPrediction(module(features), label_ids))["acc1"])
    return float(sum(top1_acc) / len(top1_acc))


# %% [markdown]
# ### Now lets predict using the trained model.
#
# Once the model is trained we want to compute metrics on how we did. Also once the model is deployed we may want to run predictions in the online server. This method can be invoked in multiple contexts and should do a singular function, **how to predict given the input feature.**
#

# %%
@model.predictor(cache=True, cache_version="1.0")
def predictor(module: nn.Module, features: torch.Tensor) -> dict:
    with torch.no_grad():
        probabilities = nn.functional.softmax(module(features)[0], dim=0)
    class_names = get_quickdraw_class_names()
    values, indices = torch.topk(probabilities, 3)
    return {class_names[i]: v.item() for i, v in zip(indices, values)}


# %% [markdown]
#  ## Scaling!
# One reason why you want to use UnionML is if you want to scale your training
# to a large cluster without any heavy lifting. UnionML is built on top of
# Flyte. The remote works using ``container`` images. Thus you can configure
# how the model should be stored, UnionML will also automatically build the
# docker image for you, using a Dockerfile that you provide. **project** and
# **domain** are Flyte concepts brought into UnionML to simplify management of
# multuple users

model.remote(
     registry="ghcr.io/unionai-oss",
     dockerfile="Dockerfile",
     config_file_path="config/config-remote.yaml",
     project="unionml",
     domain="development",
)

# %% [markdown]
# ### If you want to invoke this as a script

# %%
if __name__ == "__main__":
    num_classes = 2
    model.train(
        hyperparameters={"num_classes": num_classes},
        trainer_kwargs={"num_epochs": 1},
        data_dir="./data",
        max_examples_per_class=10000,
        class_limit=num_classes,
    )
