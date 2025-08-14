import datetime
import json
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.axes import Axes

from . import naming
from .utils import main_logger

logger = main_logger.getChild(__name__)


#########################
## Shared Plot Options ##
#########################

min_required_num_data: int = 1000

min_tree_size: int = 10  # Only analyze trees larger than this value.

dpi: int = 200

fontsize: float = 13
legend_fontsize: float = 13
stats_font_size: float = 13

alpha: float = 0.2
edge_alpha: float = 1.0
fill: bool = True
capsize: float = 10.0
rotate_ytick: bool = False
raw_label: bool = True
num_bins: int = 100
nospace: bool = True
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
    "blue",
    "tab:orange",
    "darkgreen",
    "red",
    "tab:purple",
    "magenta",
    "tab:brown",
    "lime",
    "mediumvioletred",
    "tab:cyan",
    "gold",
]

markers: list[str] = ["o", "v", "^", "s", "d"]


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
    # measure_key_l: list[str]
    measure_key: str

    figsize: tuple[float, float]

    plot_hist: bool
    # plot_box: bool
    plot_heatmap: bool

    heatmatp_height_ratio: float

    stats_colormap_width_ratio: float

    # legend_ncols: int

    suptitle_y: float


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

# Add random datasets.

# Correspondance between original dataset and random dataset.
dataset_random_dict: dict[str, list[str]] = {datset_key: [] for datset_key in dataset_key_l}

random_method_name_dict: dict[str, str] = {
    "yule": "Yule",
    "yule_arity": "Yule+arity",
    "yule_pos": "Yule+pos",
    "yule_arity_pos": "Yule+arity+pos",
    "uniform_pcfg": "UPCFG",
    "pcfg": "PCFG",
}

random_seed: int = 99999
# Yule models.
yule_model_key_l: list[str] = ["yule", "yule_arity", "yule_pos", "yule_arity_pos"]
for base_dataset_key in dataset_key_l:
    for model_key in yule_model_key_l:
        key = f"{model_key}_{base_dataset_key}_{random_seed}"
        dataset_key_name_dict[key] = random_method_name_dict[model_key]

        # Add corresponding random datasets.
        dataset_random_dict[base_dataset_key].append(key)

# PCFG.
pcfg_model_key_l: list[str] = ["uniform_pcfg", "pcfg"]
# pcfg_model_key_l: list[str] = ["uniform_pcfg"]
# pcfg_model_key_l: list[str] = ["pcfg"]
for base_dataset_key in dataset_key_l:
    for model_key in pcfg_model_key_l:
        key = f"{model_key}_{base_dataset_key}_{random_seed}"
        dataset_key_name_dict[key] = random_method_name_dict[model_key]

        # Add corresponding random datasets.
        dataset_random_dict[base_dataset_key].append(key)

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

measure_key_plot_base_line: dict[str, float] = {
    "colles": 0,
    "equal_weights_colles": 0,
    "rogers_j": 0,
}

preprocess_other_params: dict[str, dict[str, Any]] = {"default_setting": {}}
# gen_random_other_params: dict[str, dict[str, Any]] = {"default_setting": {}}
eval_other_params: dict[str, dict[str, Any]] = {"default_setting": {}}

# Other settings (e.g., model hyperparameters, beam search options...)
# This is useful for iterating over complex combination of settings (which cannot be done by simple for loops).
# setting_key -> specific setting dict.
plot_other_params: dict[str, dict[str, Any]] = {
    f"nat_langs_{measure_key}": {
        "preprocess_setting_key": "default_setting",
        "eval_setting_key": "default_setting",
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
        "measure_key": measure_key,
        "figsize": (8, 18),
        "plot_hist": True,
        "plot_heatmap": True,
        "heatmatp_height_ratio": 4,
        "stats_colormap_width_ratio": 0.05,
        "suptitle_y": 1.00,
    }
    for measure_key in measure_key_name_dict.keys()
}

# Add random dataset plot.
for dataset_key in dataset_key_l:
    for measure_key in measure_key_name_dict.keys():
        plot_other_params[f"random_{dataset_key}_{measure_key}"] = {
            "preprocess_setting_key": "default_setting",
            "eval_setting_key": "default_setting",
            "dataset_key_l": [
                dataset_key,
            ]
            + dataset_random_dict[dataset_key],
            "measure_key": measure_key,
            "figsize": (8, 9),
            "plot_hist": True,
            "plot_heatmap": False,
            "heatmatp_height_ratio": 4,
            "stats_colormap_width_ratio": 0.05,
            "suptitle_y": 1.00,
        }

#######################################
## Paths, Device and Script Settings ##
#######################################

base_dir: Path = Path("../tmp")

# Device setting
num_parallel: int = 8
use_gpu: bool = False
gpu_ids: list[int] = []

debug: bool = True
# debug: bool = False

# Used for log file name for each subprocess.
log_time: str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Set debug params.
if debug:
    alpha: float = 0.2
    edge_alpha: float = 1.0
    fill: bool = True
    capsize: float = 5.0
    rotate_ytick: bool = False
    raw_label: bool = True
    fontsize: float = 5
    num_bins: int = 100
    nospace: bool = True
    legend_fontsize: float = 5
    legend_out: bool = False

    markersize: float = 4

    base_dir = Path("../debug")
    # dataset_key_l = ["debug_data"]

    random_seed: int = 222

    dataset_key_l = [
        "debug_data",
        f"yule_debug_data_{random_seed}",
        f"yule_arity_debug_data_{random_seed}",
        f"yule_pos_debug_data_{random_seed}",
        f"yule_arity_pos_debug_data_{random_seed}",
        f"uniform_pcfg_debug_data_{random_seed}",
        f"pcfg_debug_data_{random_seed}",
    ]
    dataset_key_name_dict: dict[str, str] = {
        "debug_data": "Debug",
        f"yule_debug_data_{random_seed}": "Yule",
        f"yule_arity_debug_data_{random_seed}": "Yule+arity",
        f"yule_pos_debug_data_{random_seed}": "Yule+pos",
        f"yule_arity_pos_debug_data_{random_seed}": "Yule+arity+pos",
        f"uniform_pcfg_debug_data_{random_seed}": "UPCFG",
        f"pcfg_debug_data_{random_seed}": "PCFG",
    }

    min_required_num_data: int = 2

    # Script setting.

    num_parallel: int = 1
    # num_parallel: int = 3

    # use_gpu: bool = True
    use_gpu: bool = False
    gpu_ids: list[int] = []

    plot_other_params: dict[str, dict[str, Any]] = {
        "debug_plot": {
            "preprocess_setting_key": "debug_default",
            "eval_setting_key": "debug_default",
            "dataset_key_l": dataset_key_l,
            "measure_key": "colles",
            "figsize": (4, 9),
            "plot_hist": True,
            "plot_heatmap": True,
            "heatmatp_height_ratio": 3,
            "stats_colormap_width_ratio": 0.05,
            "suptitle_y": 1.00,
        },
    }

# Set fontsize and adjust subplots.
plt.rcParams["font.size"] = fontsize

if nospace:
    plt.rcParams["savefig.pad_inches"] = 0


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
    all_res: dict[str, list[float | int]] = dict()

    # First, load data.
    for dataset_key in params.dataset_key_l:
        eval_file: Path = naming.get_wholetree_eval_filepath(
            base_dir=base_dir, dataset_key=dataset_key, setting_key=params.eval_setting_key
        )

        dataset_res: list[float | int] = []

        with eval_file.open(mode="r") as f:
            cur_res_l = json.load(f)
            assert isinstance(cur_res_l, list)

            for d in cur_res_l:
                assert d["num_leaves"] >= min_tree_size
                dataset_res.append(d[params.measure_key])

        all_res[dataset_key] = dataset_res

    # Calculate and set the range.
    range_min: float | None = None
    range_max: float | None = None

    if params.measure_key in fixed_range_min:
        range_min = fixed_range_min[params.measure_key]

    if params.measure_key in fixed_range_max:
        range_max = fixed_range_max[params.measure_key]

    for dataset_key in all_res.keys():
        values = all_res[dataset_key]

        cur_min = min(values)
        cur_max = max(values)

        if range_min is None:
            range_min = cur_min

        if range_max is None:
            range_max = cur_max

        if cur_min < range_min:
            range_min = cur_min

        if cur_max > range_max:
            range_max = cur_max

    assert range_min is not None
    assert range_max is not None

    # Calculate histogram.
    # datase_key -> (normalized) histogram weights.
    hist_normalized_counts: dict[str, np.ndarray] = dict()
    hist_bins: dict[str, np.ndarray] = dict()  # This may be redundant.
    means: dict[str, float] = dict()

    for dataset_key in params.dataset_key_l:
        values: list[float] = all_res[dataset_key]

        # Check is there is enough data.
        if len(values) < min_required_num_data:
            raise Exception(f"Number of data {len(values)} is smaller than required {min_required_num_data}")

        counts, bins = np.histogram(values, bins=num_bins, range=(range_min, range_max))

        # Normalize the counts.
        normalized_counts = counts / sum(counts)
        weights = normalized_counts

        hist_normalized_counts[dataset_key] = weights
        hist_bins[dataset_key] = bins

        mean = np.mean(values)
        means[dataset_key] = mean

    # Plot

    # Sort datasets by the mean.
    mean_sorted_dataset_key_l: list[str] = [items[0] for items in sorted(means.items(), key=lambda x: x[1])]

    # Set row/colmn size.
    ncols: int = 2  # One for histogram and heatmap and the other for the statistics and colormap for heatmap.
    nrows: int = 0
    height_ratios: list[float] = []
    width_ratios: list[float] = [1.0, 1.0 * params.stats_colormap_width_ratio]
    if params.plot_hist:
        nrows += len(params.dataset_key_l)
        height_ratios += [1.0] * len(params.dataset_key_l)
    if params.plot_heatmap:
        nrows += 1
        height_ratios += [1.0 * params.heatmatp_height_ratio]

    assert nrows > 0

    # Create fig and axes.
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=params.figsize,
        sharex=False,
        sharey=False,
        squeeze=False,
        constrained_layout=True,
        dpi=dpi,
        height_ratios=height_ratios,
        width_ratios=width_ratios,
    )

    assert ncols == 2

    # Adjust.
    if manual_adjust:
        fig.subplots_adjust(
            left=adjust_left,
            right=adjust_right,
            top=adjust_top,
            bottom=adjust_bottom,
            wspace=adjust_wspace,
            hspace=adjust_hspace,
        )

    # Set markers and colors.
    # Use the same marker and color for different datasets.
    dataset_marker: dict[str, str] = {
        dataset_key: markers[i % len(markers)] for i, dataset_key in enumerate(params.dataset_key_l)
    }
    dataset_color: dict[str, str] = {
        # dataset_color: dict[str, tuple[float, float, float]] = {
        dataset_key: colors[i % len(colors)]
        for i, dataset_key in enumerate(params.dataset_key_l)
    }

    # Plot.
    fig.suptitle(measure_key_name_dict[params.measure_key], y=params.suptitle_y, verticalalignment="bottom")

    # Plot histograms and box plots.
    if params.plot_hist:
        for dataset_i, dataset_key in enumerate(mean_sorted_dataset_key_l):
            # Histogram plot.
            ax_hist = axes[dataset_i][0]
            ax_stats = axes[dataset_i][1]
            assert isinstance(ax_hist, Axes)
            assert isinstance(ax_stats, Axes)

            # Plot base line first.
            if params.measure_key in measure_key_plot_base_line:
                ax_hist.axvline(x=measure_key_plot_base_line[params.measure_key], color="gray")

            ax_hist.hist(
                x=hist_bins[dataset_key][:-1],
                bins=num_bins,
                histtype="stepfilled",
                color=dataset_color[dataset_key],
                alpha=alpha,
                weights=hist_normalized_counts[dataset_key],
            )

            ax_hist.hist(
                x=hist_bins[dataset_key][:-1],
                bins=num_bins,
                histtype="step",
                color=dataset_color[dataset_key],
                alpha=edge_alpha,
                weights=hist_normalized_counts[dataset_key],
                # To show only one legend in the figure.
                label=dataset_key_name_dict[dataset_key],
            )

            # Plot the mean.
            ax_hist.axvline(x=means[dataset_key], color=dataset_color[dataset_key], linestyle="dotted")

            # Box plot.
            ax_box = ax_hist.twinx()

            # TODO: show means?
            # bp = ax_box.boxplot(x=all_res[dataset_key], orientation="horizontal", positions=[0.5], showmeans=True)
            bp = ax_box.boxplot(x=all_res[dataset_key], orientation="horizontal", positions=[0.5], widths=0.4)

            for median in bp["medians"]:
                median.set_color("black")

            ax_box.set_yticks([])  # Do not show y-axis

            # Set legend.
            ax_hist.legend(fontsize=legend_fontsize)

            # Calculate and show statistics.
            data = all_res[dataset_key]
            mean_val = np.mean(data)
            stddev_val = np.std(all_res[dataset_key])
            skewness_val = np.mean((data - np.mean(data)) ** 3) / np.std(data) ** 3
            kurtosis_val = np.mean((data - np.mean(data)) ** 4) / np.std(data) ** 4 - 3

            stats_text: str = (
                f"Mean: {mean_val:.2f}\n"
                f"StdDev: {stddev_val:.2f}\n"
                f"Skewness: {skewness_val:.2f}\n"
                f"Kurtosis: {kurtosis_val:.2f}"
            )

            ax_stats.axis("off")
            ax_stats.text(
                0.05,
                0.95,
                stats_text,
                transform=ax_stats.transAxes,
                fontsize=stats_font_size,
                verticalalignment="top",
                horizontalalignment="left",
                # bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5),
            )

    # Plot heatmap.
    if params.plot_heatmap:
        ax_heatmap = axes[-1][0]
        ax_colorbar = axes[-1][1]
        assert isinstance(ax_heatmap, Axes)
        assert isinstance(ax_colorbar, Axes)

        # Calculate intersection matrix.
        intersection_matrix: list[list[float]] = []
        for source_dataset_key in mean_sorted_dataset_key_l:
            cur_row: list[float] = []
            for target_dataset_key in mean_sorted_dataset_key_l:
                source_normalized_counts = hist_normalized_counts[source_dataset_key]
                target_normalized_counts = hist_normalized_counts[target_dataset_key]

                intersect_area: float = calc_hist_intersec(
                    counts_l=[source_normalized_counts, target_normalized_counts]
                )
                cur_row.append(intersect_area)

            intersection_matrix.append(cur_row)

        mean_sorted_dataset_name_l: list[str] = [
            dataset_key_name_dict[dataset_key] for dataset_key in mean_sorted_dataset_key_l
        ]

        sns.heatmap(
            data=np.array(intersection_matrix),
            annot=True,
            fmt=".2f",
            xticklabels=mean_sorted_dataset_name_l,
            yticklabels=mean_sorted_dataset_name_l,
            ax=ax_heatmap,
            cbar_ax=ax_colorbar,
            # The range of histogram intersection is [0, 1].
            vmin=0.0,
            vmax=1.0,
        )

    # Save the plot.
    if not manual_adjust:
        # fig.tight_layout(pad=tightlayout_pad, w_pad=tightlayout_w_pad, h_pad=tightlayout_h_pad)
        pass

    save_filepath = naming.get_separate_dist_plot_filepath(base_dir=base_dir, setting_key=params.setting_key)
    save_filepath.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(save_filepath), bbox_inches="tight")

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
