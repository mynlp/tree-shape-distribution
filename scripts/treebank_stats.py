import datetime
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import matplotlib.pyplot as plt
import nltk
import numpy as np

from . import naming
from .utils import main_logger

logger = main_logger.getChild(__name__)


#########################
## Shared Plot Options ##
#########################

smaller_threshold: float = 0.2
larger_threshold: float = 0.8

# min_required_num_data: int = 10000
min_required_num_data: int = 1000

alpha: float = 0.8
fill: bool = True
capsize: float = 10.0
rotate_ytick: bool = False
raw_label: bool = True
fontsize: float = 26
num_bins: int = 100
nospace: bool = True
legend_fontsize: float = 16
legend_out: bool = False

markersize: float = 12.0

# manual_adjust: bool = False
manual_adjust: bool = False
adjust_left: float = 0.125
adjust_right: float = 0.9
adjust_top: float = 0.9
adjust_bottom: float = 0.1
adjust_wspace: float = 0.1
adjust_hspace: float = 0.1

# tight_layout option
tightlayout_pad: float = 0.3
tightlayout_w_pad: float = 0.3
tightlayout_h_pad: float = 0.3

# Set colors and markers to use.
colors: list[str] = [
    "magenta",
    "tab:brown",
    "tab:green",
    "gold",
    "tab:pink",
    "tab:purple",
    "tab:blue",
    "tab:orange",
    "tab:gray",
    "tab:red",
    "tab:gray",
]

markers: list[str] = ["o", "v", "^", "s", "d"]

# Set fontsize and adjust subplots.
plt.rcParams["font.size"] = fontsize

if nospace:
    plt.rcParams["savefig.pad_inches"] = 0


########################
## Parameters to Vary ##
########################


@dataclass
class ExecParams:
    ## Parameters to vary.

    # Setting name (e.g., model name, beam search options...)
    setting_key: str

    preprocess_setting_key: str

    # Datasets
    dataset_key_l: list[str]


# Datasets
dataset_key_l: list[str] = [
    "ptb",
    "ctb",
    "npcmj",
    "spmrl_french",
    "spmrl_german",
    "spmrl_korean",
    "spmrl_basque",
    "spmrl_hebrew",
    "spmrl_hungarian",
    "spmrl_polish",
    "spmrl_swedish",
]

# dataset_key -> dataset_name
dataset_key_name_dict: dict[str, str] = {
    "ptb": "English",
    "ctb": "Chinese",
    "npcmj": "Japanese",
    "spmrl_french": "French",
    "spmrl_german": "German",
    "spmrl_korean": "Korean",
    "spmrl_basque": "Basque",
    "spmrl_hebrew": "Hebrew",
    "spmrl_hungarian": "Hungarian",
    "spmrl_polish": "Polish",
    "spmrl_swedish": "Swedish",
}


preprocess_other_params: dict[str, dict[str, Any]] = {"default_setting": {}}
# gen_random_other_params: dict[str, dict[str, Any]] = {"default_setting": {}}

# Other settings (e.g., model hyperparameters, beam search options...)
# This is useful for iterating over complex combination of settings (which cannot be done by simple for loops).
# setting_key -> specific setting dict.
plot_other_params: dict[str, dict[str, Any]] = {
    "nat_langs": {
        "dataset_key_l": [
            "ptb",
            "ctb",
            "npcmj",
            "spmrl_french",
            "spmrl_german",
            "spmrl_korean",
            "spmrl_basque",
            "spmrl_hebrew",
            "spmrl_hungarian",
            "spmrl_polish",
            "spmrl_swedish",
        ],
        "preprocess_setting_key": "default_setting",
    },
}

#######################################
## Paths, Device and Script Settings ##
#######################################

base_dir: Path = Path("../tmp")

# Device setting
num_parallel: int = 2
use_gpu: bool = False
gpu_ids: list[int] = []

debug: bool = True
# debug: bool = False

# Used for log file name for each subprocess.
log_time: str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Set debug params.
if debug:
    base_dir = Path("../debug")
    # dataset_key_l = ["debug_data"]

    dataset_key_l = ["debug_data"]
    dataset_key_name_dict: dict[str, str] = {
        "debug_data": "Debug",
    }

    min_required_num_data: int = 2

    # Script setting.

    # num_parallel: int = 1
    num_parallel: int = 3

    # use_gpu: bool = True
    use_gpu: bool = False
    gpu_ids: list[int] = []

    plot_other_params: dict[str, dict[str, Any]] = {
        "debug_plot": {
            "dataset_key_l": ["debug_data", "debug_data", "debug_data"],
            "preprocess_setting_key": "debug_default",
        },
    }


###############
## Functions ##
###############


def gen_execparams() -> Generator[ExecParams, None, None]:
    """Generate parameters for exec_single by varying parameters."""

    for setting_key in plot_other_params:
        yield ExecParams(
            setting_key=setting_key,
            **plot_other_params[setting_key],
        )


def exec_single(params: ExecParams) -> None:
    """Function to be executed in parallel."""

    pid = multiprocessing.current_process().pid

    print(f"{pid=}: Start plotting for {params=}")

    # dataset_key -> list of tuple of values (length, )
    all_res: dict[str, list[tuple[int]]] = dict()

    # First, load data.
    for dataset_key in params.dataset_key_l:
        dataset_filepath: Path = naming.get_dataset_filepath(base_dir=base_dir, dataset_key=dataset_key)

        dataset_res: list[tuple[int]] = []

        # Load trees.
        with dataset_filepath.open(mode="r") as f:
            for l in f:
                t: nltk.Tree = nltk.Tree.fromstring(s=l, remove_empty_top_bracketing=True)

                # Retrieve data.
                length: int = len(t.leaves())

                cur_res = (length,)

                dataset_res.append(cur_res)

        all_res[dataset_key] = dataset_res

    # Calculate histogram intersection.

    # Set row/colmn size.
    ncols = 2 + 1  # +1 is for the left-most part of the table for measure name and the other for values.

    # Plot.
    table_string_l: list[str] = []
    table_string_l.append(r"\begin{table}[t]")
    table_string_l.append(r"\centering")
    table_string_l.append(r"\begin{tabular}{" + "c" * ncols + r"}")

    # Add the header.
    header: str = " & Number of Data & Number of Leaves" + r" \\"
    table_string_l.append(header)
    table_string_l.append(r"\hline\hline")

    skipped_config: set[str] = set()

    for dataset_key in params.dataset_key_l:
        values = all_res[dataset_key]

        num_data: int = len(values)
        num_leaves_l: list[int] = []

        for val in values:
            # Unpack.
            (num_leaves,) = val

            num_leaves_l.append(num_leaves)

        mean_num_leaves = np.mean(num_leaves_l)
        stddev_num_leaves = np.std(num_leaves_l)

        # Format table line.
        table_line_str = (
            f"{dataset_key_name_dict[dataset_key]} & {num_data} & {mean_num_leaves:.1f} "
            + r"$\pm$"
            + f" {stddev_num_leaves:.1f}"
        )

        table_string_l.append(table_line_str + r" \\")
        table_string_l.append(r"\hline")

    table_string_l.append(r"\end{tabular}")
    table_string_l.append(r"\caption{Caption}")
    table_string_l.append(r"\label{tab:treebank_stats}")
    table_string_l.append(r"\end{table}")

    print(f"{pid=}: {skipped_config=}")

    save_filepath = naming.get_treebank_stats_filepath(base_dir=base_dir, setting_key=params.setting_key)
    save_filepath.parent.mkdir(parents=True, exist_ok=True)

    with save_filepath.open(mode="w") as f:
        f.write("\n".join(table_string_l))

    print(f"{pid=}: Finish plotting for {params=}")


def main():
    logger.info("Start plot!!!")

    with multiprocessing.Pool(processes=num_parallel) as executer:
        # Pass the seed and hparams.
        executer.map(
            exec_single,
            gen_execparams(),
        )

    logger.info("Finish plot!!!")


if __name__ == "__main__":
    main()
