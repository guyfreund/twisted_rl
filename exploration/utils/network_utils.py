import platform
import traceback
from typing import Iterator, Optional, Union

import numpy as np
import torch
from torch import Tensor, nn, device
from torch.nn import Parameter


def to_torch(batch, dtype=torch.float32, device=None):
    if not isinstance(batch, Tensor):
        if not isinstance(batch, np.ndarray):
            batch = np.array(batch)
        batch = torch.tensor(batch, dtype=dtype, device=device)
    if device is not None:
        batch = batch.to(device)
    return batch


def process_batch(batch, device=None):
    if isinstance(batch, np.ndarray):
        start_goal = batch
    else:
        try:
            start_goal = np.array([np.concatenate((b['observation'], b['desired_goal'])) for b in batch])
        except ValueError as e:
            for i in batch:
                print(i)
            traceback.print_exception(type(e), e, e.__traceback__)
            raise e
    batch = to_torch(start_goal, device=device)
    if len(batch.shape) == 1:
        batch = torch.unsqueeze(batch, 0)
    return batch


def get_layers(sizes, dropout_prob=0.0, activation=nn.ReLU, layer_norm=False, return_packed=True):
    l = []
    for i in range(len(sizes) - 1):
        l.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            if dropout_prob > 0.:
                l.append(nn.Dropout(p=dropout_prob))
            if layer_norm:
                l.append(nn.LayerNorm(sizes[i + 1]))
            l.append(activation())

    if return_packed:
        return nn.Sequential(*l)
    else:
        return l


def generic_freeze(net, frozen, freeze_layers, net_name) -> bool:
    if not frozen and freeze_layers > 0:
        print(f'Freezing first {freeze_layers} layers of the network {net_name}')
        for i, layer in enumerate(net):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
    return True  # frozen


def generic_unfreeze(net, frozen, net_name) -> bool:
    if frozen:
        print(f'Unfreezing ALL layers of the network {net_name}')
        for layer in net:
            for param in layer.parameters():
                param.requires_grad = True
    return False  # unfrozen


def compute_gradients(parameters: Iterator[Parameter]):
    return sum([p.grad.data.norm(2) ** 2 for p in parameters]) ** 0.5


def clip_gradients_if_required(parameters: Iterator[Parameter], max_grad_norm: Optional[float] = None):
    if max_grad_norm is not None and max_grad_norm > 0:
        return nn.utils.clip_grad_norm_(parameters, max_grad_norm)
    else:
        return compute_gradients(parameters)


def get_device(string: bool = False, to_print: bool = False) -> Union[device, str]:
    system = platform.system()
    device_str = "cpu"
    using_gpu = False

    if system == "Darwin":
        gpu_available = torch.backends.mps.is_available() or torch.backends.mps.is_built()
        if gpu_available:
            device_str = "mps"
            using_gpu = True
    else:
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            device_index = torch.cuda.current_device()
            device_str = f"cuda:{device_index}"
            using_gpu = True

    if to_print:
        print(f"System: {system}")
        print(f"GPU available: {gpu_available}")
        if not using_gpu:
            print("Using CPU")
        else:
            print(f"Using GPU: {device_str}")

    return torch.device(device_str) if not string else device_str
