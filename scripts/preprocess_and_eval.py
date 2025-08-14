import datetime
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

from . import naming
from .utils import get_thread_id, main_logger, thread_log, thread_run_subprocess

logger = main_logger.getChild(__name__)

########################
## Default Parameters ##
########################


@dataclass
class PreprocessExecParams:
    ## Parameters to vary.

    setting_key: str

    # Dataset
    dataset_key: str


@dataclass
class GenRandomExecParams:
    ## Parameters to vary.

    setting_key: str

    # Dataset
    dataset_key: str


@dataclass
class EvalExecParams:
    ## Parameters to vary.

    setting_key: str

    # Dataset
    dataset_key: str


########################
## Parameters to Vary ##
########################

# Datasets.
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

# Set the base directory where all datasets are stored.
data_base_dir: str = "/home/$USER"  # local

treebank_source_dir: dict[str, str] = {
    "ptb": data_base_dir + "/resources/PTB3/treebank_3/parsed/mrg/wsj/",
    "ctb": data_base_dir + "/resources/ctb5.1_507K/data/bracketed/",
    "npcmj": data_base_dir + "/resources/NPCMJ/npcmj/",
    "kortb": data_base_dir + "/resources/KoreanTreebank2/data/Ver2.0/",
    "ftb": data_base_dir + "/resources/FrenchTreebank/frenchTreebank/tigerXML/ptb/",
    "spmrl_basque": data_base_dir + "/resources/SPMRL/SPMRL_SHARED_2014_NO_ARABIC/BASQUE_SPMRL/gold/ptb/",
    "spmrl_french": data_base_dir + "/resources/SPMRL/SPMRL_SHARED_2014_NO_ARABIC/FRENCH_SPMRL/gold/ptb/",
    "spmrl_german": data_base_dir + "/resources/SPMRL/SPMRL_SHARED_2014_NO_ARABIC/GERMAN_SPMRL/gold/ptb/",
    "spmrl_hebrew": data_base_dir + "/resources/SPMRL/SPMRL_SHARED_2014_NO_ARABIC/HEBREW_SPMRL/gold/ptb/",
    "spmrl_hungarian": data_base_dir + "/resources/SPMRL/SPMRL_SHARED_2014_NO_ARABIC/HUNGARIAN_SPMRL/gold/ptb/",
    "spmrl_korean": data_base_dir + "/resources/SPMRL/SPMRL_SHARED_2014_NO_ARABIC/KOREAN_SPMRL/gold/ptb/",
    "spmrl_polish": data_base_dir + "/resources/SPMRL/SPMRL_SHARED_2014_NO_ARABIC/POLISH_SPMRL/gold/ptb/",
    "spmrl_swedish": data_base_dir + "/resources/SPMRL/SPMRL_SHARED_2014_NO_ARABIC/SWEDISH_SPMRL/gold/ptb/",
}

min_num_leaves: int = 10

# Random datasets.
random_dataset_key_l: list[str] = []
# random_method_dict: dict[str, str] = dict()
random_params_dict: dict[str, dict[str, Any]] = dict()

random_num_data: int = 10000
random_num_iter: int = 10
random_seed: int = 99999


# Yule models.
yule_model_key_l: list[str] = ["yule", "yule_arity", "yule_pos", "yule_arity_pos"]
for base_dataset_key in dataset_key_l:
    for model_key in yule_model_key_l:
        key = f"{model_key}_{base_dataset_key}_{random_seed}"
        random_dataset_key_l.append(key)
        random_params_dict[key] = {
            "base_dataset_key": base_dataset_key,
            "method": "yule_model",
            "multi_dist_arity": "arity" in model_key,
            "multi_dist_pos": "pos" in model_key,
        }

# PCFG.
pcfg_model_key_l: list[str] = ["uniform_pcfg", "pcfg"]
# pcfg_model_key_l: list[str] = ["uniform_pcfg"]
# pcfg_model_key_l: list[str] = ["pcfg"]
for base_dataset_key in dataset_key_l:
    for model_key in pcfg_model_key_l:
        key = f"{model_key}_{base_dataset_key}_{random_seed}"
        random_dataset_key_l.append(key)
        random_params_dict[key] = {
            "base_dataset_key": base_dataset_key,
            "method": "pcfg",
            "uniform": True if model_key.startswith("uniform") else False,
        }

# Datasets to evaluate.
eval_dataset_key_l: list[str] = dataset_key_l + random_dataset_key_l

# Other settings (e.g., model hyperparameters, beam search options...)
# This is useful for iterating over complex combination of settings (which cannot be done by simple for loops).
# setting_key -> specific setting dict.
preprocess_other_params: dict[str, dict[str, Any]] = {"default_setting": {}}

gen_random_other_params: dict[str, dict[str, Any]] = {"default_setting": {}}

eval_other_params: dict[str, dict[str, Any]] = {"default_setting": {}}

#######################################
## Paths, Device and Script Settings ##
#######################################

base_dir: Path = Path("../tmp")

# Device setting
# num_parallel: int = 1
# num_parallel: int = 2
# num_parallel: int = 3
num_parallel: int = 8
use_gpu: bool = False
gpu_ids: list[int] = []

force_get_treebank: bool = False
# force_get_treebank: bool = True
force_gen_random: bool = False
# force_gen_random: bool = True
force_calc_measures: bool = False
# force_calc_measures: bool  = True

skip_gen_random: bool = False
# skip_gen_random: bool = True
skip_calc_measures: bool = False
# skip_calc_measures: bool = True


debug: bool = True
# debug: bool = False

# Used for log file name for each subprocess.
log_time: str = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

# Set debug params.
if debug:
    base_dir = Path("../debug")
    dataset_key_l = ["debug_data"]

    treebank_source_dir: dict[str, str] = {"debug_data": "./debug_data/"}

    # Random datasets.
    random_dataset_key_l: list[str] = []
    random_params_dict: dict[str, dict[str, Any]] = dict()

    random_num_data: int = 10
    random_num_iter: int = 2
    random_seed: int = 222

    # Yule model.
    yule_model_key_l: list[str] = ["yule", "yule_arity", "yule_pos", "yule_arity_pos"]
    for base_dataset_key in dataset_key_l:
        for model_key in yule_model_key_l:
            key = f"{model_key}_{base_dataset_key}_{random_seed}"
            random_dataset_key_l.append(key)
            random_params_dict[key] = {
                "base_dataset_key": base_dataset_key,
                "method": "yule_model",
                "multi_dist_arity": "arity" in model_key,
                "multi_dist_pos": "pos" in model_key,
            }

    # PCFG.
    pcfg_model_key_l: list[str] = ["uniform_pcfg", "pcfg"]
    # PCFG.
    for base_dataset_key in dataset_key_l:
        for model_key in pcfg_model_key_l:
            key = f"{model_key}_{base_dataset_key}_{random_seed}"
            random_dataset_key_l.append(key)
            random_params_dict[key] = {
                "base_dataset_key": base_dataset_key,
                "method": "pcfg",
                "uniform": True if model_key.startswith("uniform") else False,
            }

    eval_dataset_key_l: list[str] = dataset_key_l + random_dataset_key_l

    # num_parallel: int = 1
    num_parallel: int = 3
    # num_parallel: int = 3

    # use_gpu: bool = True
    use_gpu: bool = False
    gpu_ids: list[int] = []

    preprocess_other_params: dict[str, dict[str, Any]] = {"debug_default": {}}
    gen_random_other_params: dict[str, dict[str, Any]] = {"debug_default": {}}
    eval_other_params: dict[str, dict[str, Any]] = {"debug_default": {}}

###############
## Functions ##
###############


def preprocess_gen_execparams() -> Generator[PreprocessExecParams, None, None]:
    """Generate parameters for exec_single by varying parameters."""

    for dataset_key in dataset_key_l:
        for preprocess_setting_key in preprocess_other_params:
            yield PreprocessExecParams(
                dataset_key=dataset_key,
                setting_key=preprocess_setting_key,
                **preprocess_other_params[preprocess_setting_key],
            )


def gen_random_gen_execparams() -> Generator[GenRandomExecParams, None, None]:
    """Generate parameters for exec_single by varying parameters."""

    for dataset_key in random_dataset_key_l:
        for gen_random_setting_key in gen_random_other_params:
            yield GenRandomExecParams(
                dataset_key=dataset_key,
                setting_key=gen_random_setting_key,
                **gen_random_other_params[gen_random_setting_key],
            )


def eval_gen_execparams() -> Generator[EvalExecParams, None, None]:
    """Generate parameters for exec_single by varying parameters."""

    for dataset_key in eval_dataset_key_l:
        for eval_setting_key in eval_other_params:
            yield EvalExecParams(
                dataset_key=dataset_key,
                setting_key=eval_setting_key,
                **eval_other_params[eval_setting_key],
            )


def preprocess_exec_single(params: PreprocessExecParams) -> None:
    """Function to be executed in parallel."""

    thread_log(f"Execution for {params=}")

    # List of commands to execute
    commands: list[str] = []
    steps: list[str] = []  # Just for logging.

    ## Set device; gpu_id is None if the device to be used is not gpu.
    gpu_id = None
    thread_id: int = get_thread_id()
    if use_gpu and (thread_id < len(gpu_ids)):
        gpu_id = gpu_ids[thread_id]

    # gpu_option: list[str] = (
    #    ["CUDA_VISIBLE_DEVICES={}".format(gpu_id)] if gpu_id is not None else ["CUDA_VISIBLE_DEVICES=-1"]
    # )

    ## We only use one gpu, and the gpu count always start from 0 (regardless of actual gpu id).
    # device_flag: list[str] = ["--device cuda", "--gpu 0"] if gpu_id is not None else ["--device cpu"]

    #########################
    ## Supervised Training ##
    #########################

    # First, check if the result file already exists or not.
    # If the result file alredy exists, the execution is skipped unless train_force_update is True.

    preprocess_output_file: Path = naming.get_dataset_filepath(
        base_dir=base_dir,
        dataset_key=params.dataset_key,
    )

    if force_get_treebank or (not preprocess_output_file.exists()):
        # Make the directories just in case.
        preprocess_output_file.parent.mkdir(parents=True, exist_ok=True)

        source_data_dir: Path = Path(treebank_source_dir[params.dataset_key])

        get_treebank_command_str: str = " ".join(
            [
                "python get_treebank.py",
                # Files.
                f"--treebank_key {params.dataset_key}",
                f"--output_filepath {preprocess_output_file.resolve()}",
                f"--source_data_dir {source_data_dir.resolve()}",
                f"--min_num_leaves {min_num_leaves}",
            ]
        )

        commands.append(get_treebank_command_str)
        steps.append("get_treebank")
    else:
        thread_log(f"Skip get_treebank for {params}")

    # Execute the commands.
    if len(commands) == 0:
        thread_log(f"Skip execution for {params}")
    else:
        log_file: Path = naming.get_get_treebank_log_file(
            base_dir=base_dir,
            dataset_key=params.dataset_key,
            setting_key=params.setting_key,
            log_time=log_time,
        )
        log_file.parent.mkdir(parents=True, exist_ok=True)

        step_str: str = ", ".join(steps)
        command_str: str = "; ".join(["set -e"] + commands)

        thread_log(f"GPU id: {gpu_id}; Start {step_str} for {params}")

        thread_run_subprocess(command_str=command_str, log_file=log_file)

        thread_log(f"GPU id: {gpu_id}; End {step_str} for {params}")


def gen_random_exec_single(params: PreprocessExecParams) -> None:
    """Function to be executed in parallel."""

    thread_log(f"Execution for {params=}")

    # List of commands to execute
    commands: list[str] = []
    steps: list[str] = []  # Just for logging.

    ## Set device; gpu_id is None if the device to be used is not gpu.
    gpu_id = None
    thread_id: int = get_thread_id()
    if use_gpu and (thread_id < len(gpu_ids)):
        gpu_id = gpu_ids[thread_id]

    # gpu_option: list[str] = (
    #    ["CUDA_VISIBLE_DEVICES={}".format(gpu_id)] if gpu_id is not None else ["CUDA_VISIBLE_DEVICES=-1"]
    # )

    ## We only use one gpu, and the gpu count always start from 0 (regardless of actual gpu id).
    # device_flag: list[str] = ["--device cuda", "--gpu 0"] if gpu_id is not None else ["--device cpu"]

    #########################
    ## Supervised Training ##
    #########################

    # First, check if the result file already exists or not.
    # If the result file alredy exists, the execution is skipped unless train_force_update is True.

    # Use the same naming as the normal datasets.
    output_file: Path = naming.get_dataset_filepath(
        base_dir=base_dir,
        dataset_key=params.dataset_key,
    )

    if (not skip_gen_random) and (force_gen_random or force_get_treebank or (not output_file.exists())):
        # Make the directories just in case.
        output_file.parent.mkdir(parents=True, exist_ok=True)

        base_dataset_filepath: Path = naming.get_dataset_filepath(
            base_dir=base_dir,
            dataset_key=random_params_dict[params.dataset_key]["base_dataset_key"],
        )
        if random_params_dict[params.dataset_key]["method"] == "yule_model":
            multi_dist_arity: str = random_params_dict[params.dataset_key]["multi_dist_arity"]
            multi_dist_pos: str = random_params_dict[params.dataset_key]["multi_dist_pos"]

            gen_random_command_str: str = " ".join(
                [
                    "python gen_yule_trees.py",
                    f"--base_dataset_filepath {base_dataset_filepath.resolve()}",
                    f"--output_filepath {output_file.resolve()}",
                    f"--multi_dist_arity {multi_dist_arity}",
                    f"--multi_dist_pos {multi_dist_pos}",
                    f"--num_data {random_num_data}",
                    f"--num_iter {random_num_iter}",
                    f"--min_num_leaves {min_num_leaves}",
                    f"--random_seed {random_seed}",
                ]
            )
        elif random_params_dict[params.dataset_key]["method"] == "pcfg":
            uniform: bool = random_params_dict[params.dataset_key]["uniform"]
            gen_random_command_str: str = " ".join(
                [
                    "python gen_pcfg_trees.py",
                    f"--base_dataset_filepath {base_dataset_filepath.resolve()}",
                    f"--output_filepath {output_file.resolve()}",
                    f"--uniform {uniform}",
                    f"--num_data {random_num_data}",
                    f"--min_num_leaves {min_num_leaves}",
                    f"--random_seed {random_seed}",
                ]
            )
        else:
            raise Exception(
                f"No such method for generating random trees: {random_params_dict[params.dataset_key]['method']}"
            )

        commands.append(gen_random_command_str)
        steps.append("gen_random_trees")
    else:
        thread_log(f"Skip gen_random_trees for {params}")

    # Execute the commands.
    if len(commands) == 0:
        thread_log(f"Skip execution for {params}")
    else:
        log_file: Path = naming.get_gen_random_log_file(
            base_dir=base_dir,
            dataset_key=params.dataset_key,
            setting_key=params.setting_key,
            log_time=log_time,
        )
        log_file.parent.mkdir(parents=True, exist_ok=True)

        step_str: str = ", ".join(steps)
        command_str: str = "; ".join(["set -e"] + commands)

        thread_log(f"GPU id: {gpu_id}; Start {step_str} for {params}")

        thread_run_subprocess(command_str=command_str, log_file=log_file)

        thread_log(f"GPU id: {gpu_id}; End {step_str} for {params}")


def eval_exec_single(params: EvalExecParams) -> None:
    """Function to be executed in parallel."""

    thread_log(f"Execution for {params=}")

    # List of commands to execute
    commands: list[str] = []
    steps: list[str] = []  # Just for logging.

    ## Set device; gpu_id is None if the device to be used is not gpu.
    gpu_id = None
    thread_id: int = get_thread_id()
    if use_gpu and (thread_id < len(gpu_ids)):
        gpu_id = gpu_ids[thread_id]

    # gpu_option: list[str] = (
    #    ["CUDA_VISIBLE_DEVICES={}".format(gpu_id)] if gpu_id is not None else ["CUDA_VISIBLE_DEVICES=-1"]
    # )

    ## We only use one gpu, and the gpu count always start from 0 (regardless of actual gpu id).
    # device_flag: list[str] = ["--device cuda", "--gpu 0"] if gpu_id is not None else ["--device cpu"]

    ###############
    ## Inference ##
    ###############

    # First, check if the result file already exists or not.
    # If the result file alredy exists, the execution is skipped unless beam_force_update is True.

    wholetree_output_file: Path = naming.get_wholetree_eval_filepath(
        base_dir=base_dir,
        dataset_key=params.dataset_key,
        setting_key=params.setting_key,
    )

    # subtree_output_file: Path = naming.get_subtree_eval_filepath(
    #    base_dir=base_dir,
    #    dataset_key=params.dataset_key,
    #    setting_key=params.setting_key,
    # )

    if (not skip_calc_measures) and (
        force_calc_measures or force_get_treebank or (not wholetree_output_file.exists())
        # or (not subtree_output_file.exists())
    ):
        # Make the directories just in case.
        wholetree_output_file.parent.mkdir(parents=True, exist_ok=True)
        # subtree_output_file.parent.mkdir(parents=True, exist_ok=True)

        dataset_filepath: Path = naming.get_dataset_filepath(base_dir=base_dir, dataset_key=params.dataset_key)

        eval_command_str: str = " ".join(
            [
                "python calc_measures.py",
                f"--dataset_filepath {dataset_filepath.resolve()}",
                f"--wholetree_output_filepath {wholetree_output_file.resolve()}",
                # f"--subtree_output_filepath {subtree_output_file.resolve()}",
            ]
        )

        commands.append(eval_command_str)
        steps.append("calc_measure")
    else:
        thread_log(f"Skip eval for {params}")

    # Execute the commands.
    if len(commands) == 0:
        thread_log(f"Skip execution for {params}")
    else:
        log_file: Path = naming.get_eval_log_file(
            base_dir=base_dir,
            dataset_key=params.dataset_key,
            setting_key=params.setting_key,
            log_time=log_time,
        )
        log_file.parent.mkdir(parents=True, exist_ok=True)

        step_str: str = ", ".join(steps)
        command_str: str = "; ".join(["set -e"] + commands)

        thread_log(f"GPU id: {gpu_id}; Start {step_str} for {params}")

        thread_run_subprocess(command_str=command_str, log_file=log_file)

        thread_log(f"GPU id: {gpu_id}; End {step_str} for {params}")


def main():
    logger.info("Start preprocess and eval!!!")

    # with ThreadPoolExecutor(max_workers=num_parallel) as executer:
    with ThreadPoolExecutor(max_workers=num_parallel, thread_name_prefix="Thread") as executer:
        # First train
        executer.map(
            preprocess_exec_single,
            preprocess_gen_execparams(),
        )

    # with ThreadPoolExecutor(max_workers=num_parallel) as executer:
    with ThreadPoolExecutor(max_workers=num_parallel, thread_name_prefix="Thread") as executer:
        # First train
        executer.map(
            gen_random_exec_single,
            gen_random_gen_execparams(),
        )

    # with ThreadPoolExecutor(max_workers=num_parallel) as executer:
    with ThreadPoolExecutor(max_workers=num_parallel, thread_name_prefix="Thread") as executer:
        # Next evaluate.
        executer.map(
            eval_exec_single,
            eval_gen_execparams(),
        )

    logger.info("Finish preprocess and eval!!!")


if __name__ == "__main__":
    main()
