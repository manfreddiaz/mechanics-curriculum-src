import functools
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

def eval_model(
    model: torch.nn.Module,
    envs: DataLoader,
    num_steps: int,
    device: torch.device,
    metric: str = "accuracy"
):
    # TODO: config num classes
    metric = MulticlassAccuracy(num_classes=10, average="micro")
    metric.to(device)
    with torch.no_grad():
        for step in range(num_steps):
            for data in envs:
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                metric(outputs, labels)

    return [metric.compute().item()]


def make_evaluator(
    id: str,
    metric: str
):
    return functools.partial(
        eval_model,
        metric=metric
    )

