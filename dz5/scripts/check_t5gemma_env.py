from __future__ import annotations

import torch
from transformers import AutoConfig

import datasets
import peft
import transformers


def main() -> None:
    print("torch", torch.__version__)
    print("cuda", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("gpu", torch.cuda.get_device_name(0))
        print("capability", torch.cuda.get_device_capability(0))
    print("transformers", transformers.__version__)
    print("datasets", datasets.__version__)
    print("peft", peft.__version__)
    cfg = AutoConfig.from_pretrained("google/t5gemma-2-270m-270m")
    print("model_type", cfg.model_type)
    print("config_class", type(cfg).__name__)


if __name__ == "__main__":
    main()
