import dataclasses
import json
from pathlib import Path

import myutils
import nltk
import tree_shape_measure
from data import WholeTreeResults
from mylogger import main_logger
from tree_shape_measure import (
    NodeIndex,
)

logger = main_logger.getChild(__name__)


def calc_wholetree_measures(t: nltk.Tree) -> WholeTreeResults:
    root_node_index: NodeIndex = ()

    return WholeTreeResults(
        fixed_aspect_ratio=tree_shape_measure.fixed_aspect_ratio(t=t, node_index=root_node_index),
        unary_collapsed_aspect_ratio=tree_shape_measure.unary_collapsed_aspect_ratio(t=t, node_index=root_node_index),
        aspect_ratio=tree_shape_measure.aspect_ratio(t=t, node_index=root_node_index),
        mean_degree=tree_shape_measure.mean_degree(t=t, node_index=root_node_index),
        max_center_emb=tree_shape_measure.max_center_embedding(t=t, node_index=root_node_index),
        normalized_max_center_emb=tree_shape_measure.normalized_max_center_embedding(t=t, node_index=root_node_index),
        phrase_max_center_emb=tree_shape_measure.phrase_max_center_embedding(t=t, node_index=root_node_index),
        phrase_normalized_max_center_emb=tree_shape_measure.phrase_normalized_max_center_embedding(
            t=t, node_index=root_node_index
        ),
        height=tree_shape_measure.node_height(t=t, node_index=root_node_index),
        num_leaves=tree_shape_measure.num_leaves_inside(t=t, node_index=root_node_index),
        colles=tree_shape_measure.relative_corrected_colles_index(t=t, node_index=root_node_index),
        equal_weights_colles=tree_shape_measure.relative_equal_weights_corrected_colles_index(
            t=t, node_index=root_node_index
        ),
        rogers_j=tree_shape_measure.relative_rogers_j_index(t=t, node_index=root_node_index),
    )


def calc_measures(
    dataset_filepath: Path,
    wholetree_output_filepath: Path,
):
    wholetree_results: list[dict[str, float | int]] = []

    with dataset_filepath.open(mode="r") as f:
        # Process line by line.
        for l in f:
            t: nltk.Tree = nltk.Tree.fromstring(s=l, remove_empty_top_bracketing=True)

            # Calculate measures for the whole tree.
            cur_wholetree_res: WholeTreeResults = calc_wholetree_measures(t=t)
            wholetree_results.append(dataclasses.asdict(cur_wholetree_res))

    # Save the results.

    with wholetree_output_filepath.open(mode="w") as g_whole:
        json.dump(wholetree_results, g_whole)


class Args(myutils.BaseArgs):
    dataset_filepath: str
    wholetree_output_filepath: str


if __name__ == "__main__":
    # Set args.
    args = Args.parse_args()

    logger.info(f"Start calculating measures for {args.dataset_filepath}")
    calc_measures(
        dataset_filepath=Path(args.dataset_filepath),
        wholetree_output_filepath=Path(args.wholetree_output_filepath),
    )
    logger.info(f"Finish calculating measures for {args.dataset_filepath}")
