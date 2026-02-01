import os

from exploration.utils.config_utils import load_config
from exploration.utils.server_utils import main as download
from system_flow.ablations import AVAILABLE_ABLATIONS


def get_paths(data):
    results = []
    if isinstance(data, str):
        if data.startswith("/home/g/guyfreund/twisted_rl"):
            results.append(data)
    elif isinstance(data, dict):
        for key, value in data.items():
            results.extend(get_paths(value))
    elif isinstance(data, list):
        for item in data:
            results.extend(get_paths(item))
    return results


cfg = load_config(path="system_flow/config/twisted_evaluation.yml")
override = False
model_paths = [os.path.dirname(path) for path in get_paths(cfg)]
ablations_paths = [os.path.join('/home/g/guyfreund/twisted_rl', item) for sublist in [ablation.paths() for ablation in AVAILABLE_ABLATIONS] for item in sublist]

download(
    override=override,
    local_server='mac',
    remote_server='server',
    remote_paths=model_paths,
    exclude_extensions=['.pkl']
)

download(
    override=override,
    local_server='mac',
    remote_server='server',
    remote_paths=ablations_paths,
)