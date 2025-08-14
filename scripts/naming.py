from pathlib import Path

################
## Preprocess ##
################


def get_dataset_filepath(base_dir: Path, dataset_key: str) -> Path:
    return Path.joinpath(base_dir, "dataset", dataset_key, f"{dataset_key}.tree")


##########
## Eval ##
##########


def get_wholetree_eval_filepath(base_dir: Path, dataset_key: str, setting_key: str) -> Path:
    return Path.joinpath(base_dir, "eval", dataset_key, setting_key, "whole_tree_res.json")


##############
## Analysis ##
##############


def get_hist_compare_average_filepath(base_dir: Path, setting_key: str) -> Path:
    return Path.joinpath(base_dir, "analysis", "hist_compare_average", f"{setting_key}.tex")


def get_hist_compare_multi_filepath(base_dir: Path, setting_key: str) -> Path:
    return Path.joinpath(base_dir, "analysis", "hist_compare_multi", f"{setting_key}.tex")


def get_hist_intersec_wholetree_filepath(base_dir: Path, setting_key: str) -> Path:
    return Path.joinpath(base_dir, "analysis", "hist_intersec_wholetree", f"{setting_key}.tex")


def get_separate_dist_plot_filepath(base_dir: Path, setting_key: str) -> Path:
    return Path.joinpath(base_dir, "analysis", "separate_dist", f"{setting_key}.png")


def get_treebank_stats_filepath(base_dir: Path, setting_key: str) -> Path:
    return Path.joinpath(base_dir, "analysis", "treebank_stats", f"{setting_key}.tex")


#########
## Log ##
#########


def get_get_treebank_log_file(base_dir: Path, dataset_key: str, setting_key: str, log_time: str) -> Path:
    return Path.joinpath(base_dir, "log", "get_treebank", dataset_key, setting_key, f"{log_time}.log")


def get_gen_random_log_file(base_dir: Path, dataset_key: str, setting_key: str, log_time: str) -> Path:
    return Path.joinpath(base_dir, "log", "gen_random", dataset_key, setting_key, f"{log_time}.log")


def get_eval_log_file(base_dir: Path, dataset_key: str, setting_key: str, log_time: str) -> Path:
    return Path.joinpath(base_dir, "log", "eval", dataset_key, setting_key, f"{log_time}.log")
