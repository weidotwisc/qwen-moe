import os
import re
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


# [stacked MoE experts, C5-C9] HF ships one weight per expert; route
# experts.{e}.{gate,up,down}_proj.weight -> stacked param {...mlp}.w_{gate,up,down}
# slice [e]. Mirrors vLLM: the loader must parse the expert index, which the
# static packed_modules_mapping cannot encode.
_EXPERT_RE = re.compile(r"(.*)\.experts\.(\d+)\.(gate|up|down)_proj\.weight$")


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                m = _EXPERT_RE.match(weight_name)
                if m is not None:
                    prefix, expert_id, proj = m.group(1), int(m.group(2)), m.group(3)
                    param = model.get_parameter(f"{prefix}.w_{proj}")
                    param.weight_loader(param, f.get_tensor(weight_name), expert_id)
                    continue
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        param_name = weight_name.replace(k, v)
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
