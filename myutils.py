import argparse
import random

import numpy as np
import pydantic

# import torch

#############
## General ##
#############


def set_random_seed(random_seed: int):
    """Set the random seeds to the given value."""
    random.seed(random_seed)

    # torch.manual_seed(random_seed)

    np.random.seed(random_seed)


class BaseArgs(pydantic.BaseModel):
    @classmethod
    def parse_args(cls):
        parser = argparse.ArgumentParser()
        for k in cls.model_json_schema()["properties"].keys():
            # parser.add_argument(f"-{k[0:1]}", f"--{k}")
            parser.add_argument(f"--{k}")
        return cls.model_validate(parser.parse_args().__dict__)


##############
## Specific ##
##############


def normalize_category(cat_str: str) -> str:
    if cat_str.startswith("-") and cat_str.endswith("-"):
        # Leave categories, e.g., -LRB-
        return cat_str

    # Get the left side of '-'
    cat: str = cat_str.split("-")[0]

    # Get the left side of '='
    cat: str = cat.split("=")[0]

    # Get the left side of '|'
    cat: str = cat.split("|")[0]

    # Get the left side of ';'
    # Spcific to Keyaki Treebank.
    cat: str = cat.split(";")[0]

    # Get the left side of '{'
    # Spcific to Keyaki Treebank.
    cat: str = cat.split("{")[0]

    # Specific to SPMRL
    cat: str = cat.split("##")[0]

    return cat
