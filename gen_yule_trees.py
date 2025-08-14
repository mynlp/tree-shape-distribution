"""Randomly generate trees by Yule model based on treebank statistics."""

import random
import sys
from pathlib import Path

import myutils
import nltk
from mylogger import main_logger
from tree_shape_measure import Leaf, NodeIndex, Tree

logger = main_logger.getChild(__name__)

# Dummy leaf string for the generated trees.
DUMMY_CATEGORY: str = "X"
DUMMY_LEAF: Leaf = "Y"

###########
## Utils ##
###########


def get_arity_counts(tl: list[nltk.Tree]) -> dict[int, int]:
    """
    Exclude leaves.
    No smoothing for now.
    """
    # ari -> count
    ari_counts: dict[int, int] = dict()

    for t in tl:
        # Leaves are not enumerated.
        for sub_t in t.subtrees():
            # Get the degree of the vertex.
            arity: int = len(sub_t)

            if arity not in ari_counts:
                ari_counts[arity] = 0

            ari_counts[arity] += 1

    return ari_counts


def get_length_counts(tl: list[nltk.Tree]) -> dict[int, int]:
    """No smoothing for now."""
    # length -> count
    len_counts: dict[int, int] = dict()

    for t in tl:
        length: int = len(t.leaves())

        if length not in len_counts:
            len_counts[length] = 0

        len_counts[length] += 1

    return len_counts


def get_leaf_positions(t: nltk.Tree) -> list[NodeIndex]:
    return [t.leaf_treeposition(leaf_index) for leaf_index in range(len(t.leaves()))]


def is_nary_cherry(t: Tree) -> bool:
    """Check if a given tree is an n-ary cherry."""
    if isinstance(t, Leaf):
        return False
    else:
        for child in t:
            if not isinstance(child, Leaf):
                return False
        return True


class LeafPositionKeyError(Exception):
    pass


def get_arity_and_pos_counts(
    tree_str_l: list[str], num_iter: int
) -> tuple[dict[int, dict[int, int]], dict[int, dict[int, int]]]:
    """Get the counts of replaced leaf indices and arity used in Yule model.

    The counts are calculated via somethign like inverse of Yule model.
    If the inverse process is not unique, it's done uniformly randomly."""

    # length -> arity -> count
    arity_counts: dict[int, dict[int, int]] = dict()
    # length -> position -> count
    leaf_pos_counts: dict[int, dict[int, int]] = dict()

    # Since the inverse process is random. It may be good to repeat the process for several times.
    # We need to iterate over tree string since we directly modify the tree in the inverse process.
    for _ in range(num_iter):
        for tree_str in tree_str_l:
            t = nltk.Tree.fromstring(tree_str, remove_empty_top_bracketing=True)
            # Root node is not considered.
            while not is_nary_cherry(t=t):
                cherry_root_set: set[NodeIndex] = set()

                # Enumerate the root node of n-ary cherries.
                for leaf_pos in get_leaf_positions(t):
                    parent_pos: NodeIndex = leaf_pos[:-1]

                    if is_nary_cherry(t[parent_pos]):
                        cherry_root_set.add(parent_pos)

                # Randomly pick one cherry.
                # Sort the set just to avoid unintended randomness.
                cherry_root: NodeIndex = random.choice(sorted(list(cherry_root_set)))

                cherry_arity: int = len(t[cherry_root])

                # Replace the cherry root with a single leaf (inverse of Yule model).
                t[cherry_root] = DUMMY_LEAF

                # Find the corresponding leaf index.
                leaf_ids: list[int] = [i for i, p in enumerate(get_leaf_positions(t)) if p == cherry_root]
                # Just to make sure.
                assert len(leaf_ids) == 1

                leaf_id: int = leaf_ids[0]

                # Update arity counts.
                # Only update the counts for the current length.
                length: int = len(t.leaves())
                if length not in arity_counts:
                    arity_counts[length] = dict()

                if cherry_arity not in arity_counts[length]:
                    arity_counts[length][cherry_arity] = 0

                arity_counts[length][cherry_arity] += 1

                # Update leaf position counts counts.
                # Only update the counts for the current length.
                if length not in leaf_pos_counts:
                    leaf_pos_counts[length] = dict()

                if leaf_id not in leaf_pos_counts[length]:
                    leaf_pos_counts[length][leaf_id] = 0

                leaf_pos_counts[length][leaf_id] += 1

            # t is an n-ary cherry.
            final_arity: int = len(t)

            # Note that Yule model starts from single leaf node.
            if 1 not in arity_counts:
                arity_counts[1] = dict()
            if final_arity not in arity_counts[1]:
                arity_counts[1][final_arity] = 0

            arity_counts[1][final_arity] += 1

    return arity_counts, leaf_pos_counts


class NoStatisticsError(Exception):
    pass


def sample_single_yule_tree(
    length_counts: dict[int, int],
    arity_counts: dict[int, dict[int, int]],
    leaf_position_counts: dict[int, dict[int, int]],
) -> nltk.Tree:
    # First, sample length.
    length_keys: list[int] = sorted(length_counts.keys())
    length_weights: list[int] = [length_counts[l] for l in length_keys]
    length: int = random.choices(population=length_keys, weights=length_weights, k=1)[0]

    # Initialize tree by n-ary cherry.
    # Must start from cherry due to the specification of nltk.Tree; it doesn't allow single-leaf-node tree.

    # Sample arity.
    arity_keys: list[int] = sorted(arity_counts[1].keys())
    arity_weights: list[int] = [arity_counts[1][a] for a in arity_keys]

    arity: int = random.choices(population=arity_keys, weights=arity_weights, k=1)[0]

    t: nltk.Tree = nltk.Tree(DUMMY_CATEGORY, [DUMMY_LEAF for _ in range(arity)])

    # Start Yule process.

    # Loop until there are enough leaves.
    while len(t.leaves()) < length:
        cur_num_leaves: int = len(t.leaves())

        if (cur_num_leaves not in leaf_position_counts) or (cur_num_leaves not in arity_counts):
            # Restart from initialization if there is no statistical information for the length.
            # Note that cur_num_leaves may be in only one of leaf pos and arity counts when only one of multi_dist_arity or multi_dist_pos is True.
            raise NoStatisticsError

        # Sample a leaf position to replace with an n-ary cherry.
        leaf_pos_keys: list[int] = sorted(leaf_position_counts[cur_num_leaves].keys())
        leaf_pos_weights: list[int] = [leaf_position_counts[cur_num_leaves][p] for p in leaf_pos_keys]
        leaf_index: int = random.choices(population=leaf_pos_keys, weights=leaf_pos_weights, k=1)[0]

        leaf_pos: NodeIndex = t.leaf_treeposition(leaf_index)

        # Replace the selected leaf with n-ary cherry.

        # Sample cherry.
        cherry_arity_keys: list[int] = sorted(arity_counts[cur_num_leaves].keys())
        cherry_arity_weights: list[int] = [arity_counts[cur_num_leaves][a] for a in cherry_arity_keys]
        cherry_arity: int = random.choices(population=cherry_arity_keys, weights=cherry_arity_weights, k=1)[0]

        # Replace.
        t[leaf_pos] = nltk.Tree(DUMMY_CATEGORY, [DUMMY_LEAF for _ in range(cherry_arity)])

    return t


# Naive: may be inefficient.
def get_statistics(
    tree_str_l: list[str],
    tl: list[nltk.Tree],
    multi_dist_arity: bool,
    multi_dist_pos: bool,
    num_iter: int,
) -> tuple[dict[int, int], dict[int, dict[int, int]], dict[int, dict[int, int]]]:
    arity_counts, leaf_pos_counts = get_arity_and_pos_counts(tree_str_l=tree_str_l, num_iter=num_iter)

    length_counts: dict[int, int] = get_length_counts(tl)
    max_len = max(list(length_counts.keys()))

    # Overwrite depending on the option.

    if not multi_dist_arity:
        # Count arities over the whole corpus.
        corpus_arity_counts: dict[int, int] = get_arity_counts(tl=tl)

        # Always use the same counts.
        for len in range(1, max_len + 1):
            arity_counts[len] = corpus_arity_counts

    if not multi_dist_pos:
        # Always use uniform counts.
        for len in range(1, max_len + 1):
            leaf_pos_counts[len] = {pos: 1 for pos in range(len)}

    return length_counts, arity_counts, leaf_pos_counts


def gen_yule_trees(
    base_treebank: Path,
    output_filepath: Path,
    num_data: int,
    multi_dist_arity: bool,
    multi_dist_pos: bool,
    num_iter: int,
    min_num_leaves: int,
):
    # First, get the statistics from the given treebank.
    treebank: list[nltk.Tree] = []
    source_tree_str_l: list[
        str
    ] = []  # This is inefficient but necessary for calculating leaf_pos_weights by iterating the treebank for more than once.
    with base_treebank.open(mode="r") as f:
        for l in f:
            t = nltk.Tree.fromstring(l)
            treebank.append(t)
            source_tree_str_l.append(l.rstrip())

    logger.info("Start calculating statistics.")

    length_counts, arity_counts, leaf_pos_counts = get_statistics(
        tree_str_l=source_tree_str_l,
        tl=treebank,
        multi_dist_arity=multi_dist_arity,
        multi_dist_pos=multi_dist_pos,
        num_iter=num_iter,
    )

    # Sample yule trees.
    t_str_l: list[str] = []

    while len(t_str_l) < num_data:
        try:
            sampled: nltk.Tree = sample_single_yule_tree(
                length_counts=length_counts,
                arity_counts=arity_counts,
                leaf_position_counts=leaf_pos_counts,
            )
        except NoStatisticsError:
            continue

        # Format the nltk Tree to string.
        # Only collect trees that have more leaves than min_num_leaves.
        if len(sampled.leaves()) >= min_num_leaves:
            t_str: str = sampled.pformat(margin=sys.maxsize)

            t_str_l.append(t_str)

    with output_filepath.open(mode="w") as g:
        g.write("\n".join(t_str_l))


class Args(myutils.BaseArgs):
    base_dataset_filepath: str
    output_filepath: str

    multi_dist_arity: bool
    multi_dist_pos: bool

    num_data: int
    num_iter: int

    min_num_leaves: int

    random_seed: int


if __name__ == "__main__":
    # Set args.
    args = Args.parse_args()

    myutils.set_random_seed(random_seed=args.random_seed)

    logger.info("Start generating Yule trees!!!!!!")
    gen_yule_trees(
        base_treebank=Path(args.base_dataset_filepath),
        output_filepath=Path(args.output_filepath),
        num_data=args.num_data,
        multi_dist_arity=args.multi_dist_arity,
        multi_dist_pos=args.multi_dist_pos,
        num_iter=args.num_iter,
        min_num_leaves=args.min_num_leaves,
    )
    logger.info("Finish generating Yule trees!!!!!!")
