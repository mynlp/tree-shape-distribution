import datetime
import json
import multiprocessing
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any, Generator

import matplotlib.pyplot as plt
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


##############################
## Common phrase categories ##
##############################


# The range fixed in plotting.
fixed_range_min: dict[str, float] = {
    "fixed_aspect_ratio": 0,
    "aspect_ratio": 0,
    "unary_collapsed_aspect_ratio": 0,
    "mean_degree": 0,
    "max_center_emb": 0,
    "normalized_max_center_emb": 0,
    "phrase_max_center_emb": 0,
    "phrase_normalized_max_center_emb": 0,
    "height": 0,
    "colles": -1,
    "equal_weights_colles": -1,
    "rogers_j": -1,
    "num_leaves": 0,
}
fixed_range_max: dict[str, float] = {
    "fixed_aspect_ratio": 1,
    "aspect_ratio": 1,
    "unary_collapsed_aspect_ratio": 1,
    "normalized_max_center_emb": 1,
    "phrase_normalized_max_center_emb": 1,
    "colles": 1,
    "equal_weights_colles": 1,
    "rogers_j": 1,
    "span_ratio": 1,
}

########################
## Parameters to Vary ##
########################


@dataclass
class ExecParams:
    ## Parameters to vary.

    # Setting name (e.g., model name, beam search options...)
    setting_key: str

    preprocess_setting_key: str
    eval_setting_key: str

    # Datasets
    dataset_key_l: list[str]

    # Measure
    measure_key_l: list[str]


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

measure_key_name_dict: dict[str, str] = {
    "fixed_aspect_ratio": "AR",
    "phrase_normalized_max_center_emb": "NCE",
    "colles": "CC",
    "equal_weights_colles": "EWC",
    "rogers_j": "RJ",
    # "aspect_ratio": "aspect_ratio",
    # "unary_collapsed_aspect_ratio": "unary_collapsed_aspect_ratio",
    # "mean_degree": "mean_degree",
    # "max_center_emb": "max_center_emb",
    # "normalized_max_center_emb": "normalized_max_center_emb",
    # "phrase_max_center_emb": "phrase_max_center_emb",
    # "height": "height",
    # "num_leaves": "num_leaves",
}


preprocess_other_params: dict[str, dict[str, Any]] = {"default_setting": {}}
# gen_random_other_params: dict[str, dict[str, Any]] = {"default_setting": {}}
eval_other_params: dict[str, dict[str, Any]] = {"default_setting": {}}

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
        "measure_key_l": list(measure_key_name_dict.keys()),
        "preprocess_setting_key": "default_setting",
        "eval_setting_key": "default_setting",
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
            "measure_key_l": ["colles", "fixed_aspect_ratio"],
            "preprocess_setting_key": "debug_default",
            "eval_setting_key": "debug_default",
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


def calc_hist_intersec(counts_l: list[np.ndarray]) -> float:
    """Calculate the intersection area of given n histogram (normalized) counts.

    Args:
        counts_l:
            List of arrays of the same length.
    """
    intersects: np.ndarray = np.min(counts_l, axis=0)
    intersect_area: float = np.sum(intersects)
    return intersect_area


def exec_single(params: ExecParams) -> None:
    """Function to be executed in parallel."""

    pid = multiprocessing.current_process().pid

    print(f"{pid=}: Start plotting for {params=}")

    # dataset_key -> measure_key -> list of values
    all_res: dict[str, dict[str, list[float | int]]] = dict()

    # First, load data.
    for dataset_key in params.dataset_key_l:
        eval_file: Path = naming.get_wholetree_eval_filepath(
            base_dir=base_dir, dataset_key=dataset_key, setting_key=params.eval_setting_key
        )

        dataset_res: dict[str, list[float | int]] = {measure_key: [] for measure_key in params.measure_key_l}

        with eval_file.open(mode="r") as f:
            cur_res_l = json.load(f)
            assert isinstance(cur_res_l, list)

            for d in cur_res_l:
                for measure_key in params.measure_key_l:
                    dataset_res[measure_key].append(d[measure_key])

        all_res[dataset_key] = dataset_res

    # Calculate the range for each measure.
    range_min: dict[str, float] = {k: fixed_range_min[k] for k in params.measure_key_l if k in fixed_range_min}
    range_max: dict[str, float] = {k: fixed_range_max[k] for k in params.measure_key_l if k in fixed_range_max}

    for measure_key in params.measure_key_l:
        for dataset_key in all_res.keys():
            values = all_res[dataset_key][measure_key]

            cur_min = min(values)
            cur_max = max(values)

            if measure_key not in range_min:
                range_min[measure_key] = cur_min

            if measure_key not in range_max:
                range_max[measure_key] = cur_max

            if cur_min < range_min[measure_key]:
                range_min[measure_key] = cur_min

            if cur_max > range_max[measure_key]:
                range_max[measure_key] = cur_max

    # Calculate histogram intersection.

    # Set row/colmn size.
    ncols = 1 + 1  # +1 is for the left-most part of the table for measure name and the other for values.

    # Plot.
    table_string_l: list[str] = []
    table_string_l.append(r"\begin{table}[t]")
    table_string_l.append(r"\centering")
    table_string_l.append(r"\begin{tabular}{" + "c" * ncols + r"}")

    # Add the header.
    header: str = " & HI" + r" \\"
    table_string_l.append(header)
    table_string_l.append(r"\hline\hline")

    skipped_config: set[str] = set()

    # Add values for each measure.
    for measure_key in params.measure_key_l:
        table_line_str_l: list[str] = [f"{measure_key_name_dict[measure_key]}"]

        # Calculate the intersection.
        normalized_count_l: list[list[float]] = []

        tmp_dataset_count: int = 0

        for dataset_key in params.dataset_key_l:
            values: list[float] = all_res[dataset_key][measure_key]

            if len(values) < min_required_num_data:
                skipped_config.add(dataset_key)
                continue

            counts, _ = np.histogram(values, bins=num_bins, range=(range_min[measure_key], range_max[measure_key]))
            # Normalize the counts.
            normalized_counts = counts / sum(counts)

            normalized_count_l.append(normalized_counts)

            tmp_dataset_count += 1

        # Only consider cases where there are more than one dataset.
        if tmp_dataset_count > 1:
            intersect_area: float = calc_hist_intersec(counts_l=normalized_count_l)

            val_deci: Decimal = Decimal(str(intersect_area)).quantize(Decimal("0.01"), ROUND_HALF_UP)

            val_str: str = f"{val_deci:.2f}"

            # Apply hightlight.
            if intersect_area <= smaller_threshold:
                val_str = r"\textcolor{blue}{" + f"{val_str}" + "}"

            elif intersect_area >= larger_threshold:
                val_str = r"\textcolor{red}{" + f"{val_str}" + "}"

            table_line_str_l.append(val_str)
        else:
            table_line_str_l.append("-")

        table_string_l.append(" & ".join(table_line_str_l) + r" \\")
        table_string_l.append(r"\hline")

    table_string_l.append(r"\end{tabular}")
    table_string_l.append(r"\caption{Caption}")
    table_string_l.append(r"\label{tab:hist_intersec}")
    table_string_l.append(r"\end{table}")

    print(f"{pid=}: {skipped_config=}")

    save_filepath = naming.get_hist_intersec_wholetree_filepath(base_dir=base_dir, setting_key=params.setting_key)
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
