import yaml
import torch
import numpy as np
from contextlib import contextmanager

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

@contextmanager
def fix_yaml_sequence(precision=7):
    """Context manager to fix yaml sequence dumper to have numerical sequence in one line"""
    def represent_array_sequence(dumper, data, precision=precision):
        if isinstance(data, torch.Tensor):
            data = data.cpu().numpy().round(precision).tolist()
        elif isinstance(data, np.ndarray):
            data = data.round(precision).tolist()
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    # Save original representers
    original_representers = {
        list: yaml.representer.SafeRepresenter.represent_list,
        tuple: yaml.representer.SafeRepresenter.represent_list,
        np.ndarray: yaml.representer.SafeRepresenter.represent_list,
        torch.Tensor: yaml.representer.SafeRepresenter.represent_list,
    }

    # Register the custom representer
    for type in original_representers.keys():
        yaml.add_representer(type, represent_array_sequence)

    try:
        yield
    finally:
        # Restore the original representers
        for type, representer in original_representers.items():
            yaml.add_representer(type, representer)