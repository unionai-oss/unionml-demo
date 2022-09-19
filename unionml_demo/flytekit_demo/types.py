from typing import Annotated

import torch
from flytekitplugins.onnxpytorch import PyTorch2ONNX, PyTorch2ONNXConfig


PictionaryONNXModel = Annotated[
    PyTorch2ONNX,
    PyTorch2ONNXConfig(
        args=torch.randn(256, 1, 28, 28, requires_grad=True),
        export_params=True,
        opset_version=10,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        }
    ),
]
