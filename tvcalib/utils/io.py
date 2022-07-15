import yaml
import json
from typing import List
import torch


def tensor2list(d: dict):
    tensor2list_lambda = lambda x: x.detach().cpu().numpy().tolist()
    for k in d.keys():
        if isinstance(d[k], torch.Tensor):
            d[k] = tensor2list_lambda(d[k])
        if isinstance(d[k], List):
            if isinstance(d[k][0], torch.Tensor):
                d[k] = [tensor2list_lambda(x) for x in d[k]]
    return d


def write_json(json_serializable_dict, fout, indent=2):
    with open(fout, "w") as fw:
        json.dump(json_serializable_dict, fw, indent=indent)


def write_yaml(json_serializable_dict, fout):
    with open(fout, "w") as fw:
        yaml.dump(json_serializable_dict, fw, default_flow_style=False)
