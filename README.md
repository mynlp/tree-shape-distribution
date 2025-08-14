# On the Flatness, Non-linearity, and Branching Direction of Natural Language and Random Constituency Trees: Analyzing Structural Variation within and across Languages

This repository provides the experimental codes for the QUASY@SyntaxFest 2025 paper [On the Flatness, Non-linearity, and Branching Direction of Natural Language and Random Constituency Trees: Analyzing Structural Variation within and across Languages](https://aclanthology.org/2025.quasy-1.12/).

# Environment

Simply running `uv run python ...` automatically recovers the environment.

# Experiments

## Preprocess and Evaluation

`scripts/preprocess_and_eval.py` is used. Hyperparameters are directly coded in the script.
This script consists of 3 steps: 1) preprocessing of treebanks, 2) generation of random trees, and 3) calculation of tree shape measures.

Run

```
uv run python -m scripts.preprocess_and_eval
```

For full experiments, you (at least) need to modify the script and set the following variables:

```
data_base_dir: str = "PathToDirectory" # Set the directory where all datasets are placed.
debug: bool = False # Set False, the default is True.
```

Note that you can also speed up the experiment by setting the following variable:

```
num_parallel: int
```

Although there is an option for gpu, the experiments basically do not use gpus.

## Analysis

To generate Figure 3, 4, 5, 6, 8, and 9, use `scripts/plot.py`.
To generate Table 2, use `scripts/treebank_stats.py`.
To generate Table 3, use `scripts/hist_intersec_whole.py`.
To generate Table 4, use `scripts/hist_compare_average.py`.
To generate Table 4, use `scripts/hist_compare_multi.py`.

Hyperparameters are directly coded in the scripts.

Run

```
uv run python -m scripts.plot
uv run python -m scripts.treebank_stats
uv run python -m scripts.hist_intersec_whole
uv run python -m scripts.hist_compare_average
uv run python -m scripts.hist_compare_multi
```

For full experiments, you need to modify the scripts and set the following variables:

```
base_dir: Path = Path("../tmp") # Set the directory where all results will be placed.
debug: bool = False # Set False, the default is True

num_parallel: int # Number of parallel process.
```

# License

This code is distributed under the MIT License.
