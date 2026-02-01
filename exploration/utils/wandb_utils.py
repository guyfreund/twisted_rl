import wandb


def flatten_dict(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def wandb_log_fn(d: dict, wandb_run=None):
    new_d = flatten_dict(d)
    if wandb_run is not None:
        wandb_run.log(new_d)
    else:
        wandb.log(new_d)
