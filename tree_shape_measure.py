"""Calculate the tree shape measures of given trees."""

from typing import Callable, Generator, TypeAlias

import nltk
import numpy as np
from mylogger import main_logger

logger = main_logger.getChild(__name__)


#####################
## Tree Definition ##
#####################

Leaf: TypeAlias = str
Tree: TypeAlias = nltk.Tree | Leaf

NodeIndex: TypeAlias = tuple[
    int, ...
]  # Tuple of integers specifying the path from the root to the node. Each integer specifies the index of a child node relative to its siblings.

TreeShapeMeasure: TypeAlias = Callable[[Tree, NodeIndex], float]

##########
## Util ##
##########


def is_node_outside_left(base: NodeIndex, target: NodeIndex) -> bool:
    """Compute whether the target node is in the left outside of base node.

    The target node may not be at the same depth with the base node.
    """
    for i in range(len(base)):
        if base[i] > target[i]:
            return True

    return False


def is_node_outside_right(base: NodeIndex, target: NodeIndex) -> bool:
    """Compute whether the target node is in the right outside of base node.

    The target node may not be at the same depth with the base node.
    """
    for i in range(len(base)):
        if base[i] < target[i]:
            return True

    return False


def iterate_over_inner_nodes(t: Tree) -> Generator[Tree, None, None]:
    if isinstance(t, Leaf):
        return
    else:
        yield from t.subtrees()


def flip_tree(t: Tree) -> Tree:
    if isinstance(t, Leaf):
        return t
    else:
        cat: str = t.label()

        flipped_children: list[Tree] = [flip_tree(child) for child in t]

        # Reverse the order of children.
        flipped: Tree = nltk.Tree(cat, flipped_children[::-1])

        return flipped


def get_num_leaves(t: Tree) -> int:
    if isinstance(t, Leaf):
        return 1
    else:
        return len(t.leaves())


def non_leaf_node_indices(t: Tree) -> list[NodeIndex]:
    all_treepositions: set[NodeIndex] = set(t.treepositions())
    leaf_treepositions: set[NodeIndex] = set()

    for leaf_id in range(len(t.leaves())):
        leaf_treepositions.add(t.leaf_treeposition(leaf_id))

    non_leaf_treepositions = all_treepositions - leaf_treepositions
    return sorted(list(non_leaf_treepositions))


##########
## Leaf ##
##########


def num_leaves_inside(t: Tree, node_index: NodeIndex) -> int:
    if isinstance(t, Leaf):
        return 1
    else:
        inside_tree: nltk.Tree = t[node_index]
        return len(inside_tree.leaves())


def num_leaves_outside_left(t: Tree, node_index: NodeIndex) -> int:
    count: int = 0

    for i in range(len(t.leaves())):
        # Count from left.
        leaf_index: NodeIndex = t.leaf_treeposition(i)
        if is_node_outside_left(base=node_index, target=leaf_index):
            count += 1
        else:
            break
    return count


def num_leaves_outside_right(t: Tree, node_index: NodeIndex) -> int:
    count: int = 0

    num_leaves: int = len(t.leaves())

    for i in range(num_leaves):
        # Count from right.
        leaf_index: NodeIndex = t.leaf_treeposition(num_leaves - 1 - i)
        if is_node_outside_right(base=node_index, target=leaf_index):
            count += 1
        else:
            break
    return count


def num_leaves_outside(t: Tree, node_index: NodeIndex) -> int:
    outside_left: int = num_leaves_outside_left(t, node_index)
    outside_right: int = num_leaves_outside_right(t, node_index)

    return outside_left + outside_right


def span_ratio(t: Tree, node_index: NodeIndex) -> float:
    assert not isinstance(t, Leaf)

    phrase_len: int = num_leaves_inside(t, node_index)
    sentense_len: int = len(t.leaves())

    span_ratio: float = phrase_len / sentense_len
    return span_ratio


#################
## Tree Height ##
#################


def node_depth(t: Tree, node_index: NodeIndex) -> int:
    """Depth of the given node from root node."""
    return len(node_index)


def node_height(t: Tree, node_index: NodeIndex) -> int:
    """Height of phrase."""
    if isinstance(t, Leaf):
        # Leaves are at level 0.
        return 0
    else:
        phrase: Tree = t[node_index]

        # Somehow, nltk.Tree.height returns the number of nodes in the longest path.
        # So, we must subtract 1 from the nltk.Tree.height.
        level: int = phrase.height() - 1

        return level


###############
## Flatness ##
###############


def aspect_ratio(t: Tree, node_index: NodeIndex) -> float:
    if isinstance(t, Leaf):
        return 0
    else:
        phrase: Tree = t[node_index]
        num_inner_nodes: int = len(list(iterate_over_inner_nodes(phrase)))

        ratio: float = num_inner_nodes / (len(phrase.leaves()))

        return ratio


def fixed_aspect_ratio(t: Tree, node_index: NodeIndex) -> float:
    if isinstance(t, Leaf):
        return 0
    else:
        phrase: Tree = t[node_index]
        num_leaves = len(phrase.leaves())

        if num_leaves <= 2:
            return 0.0
        else:
            num_inner_nodes: int = len(list(iterate_over_inner_nodes(phrase)))

            ratio: float = (num_inner_nodes - 1) / (num_leaves - 2)

            return ratio


def unary_collapsed_aspect_ratio(t: Tree, node_index: NodeIndex) -> float:
    if isinstance(t, Leaf):
        return 0
    else:
        phrase: Tree = t[node_index]
        # Only consider inner nodes (nonterminals) that are not unary.
        num_inner_nodes: int = len(list(phrase.subtrees(lambda x: len(x) > 1)))

        ratio: float = num_inner_nodes / (len(phrase.leaves()))

        return ratio


def mean_degree(t: Tree, node_index: NodeIndex) -> float:
    if isinstance(t, Leaf):
        return 0
    else:
        phrase: Tree = t[node_index]
        degree_l: list[int] = [len(sub_t) for sub_t in iterate_over_inner_nodes(phrase)]

        num_inner_nodes = len(degree_l)

        mean_degree: float = sum(degree_l) / (num_inner_nodes)

        return mean_degree


def flatness(t: Tree, node_index: NodeIndex) -> float:
    """Basically the inverse of aspect ratio."""
    if isinstance(t, Leaf):
        return 0
    else:
        phrase: Tree = t[node_index]

        num_inner_nodes: int = len(list(iterate_over_inner_nodes(phrase)))

        ratio: float = (len(phrase.leaves())) / num_inner_nodes

        return ratio


######################
## Center Embedding ##
######################


def center_embedding_outside(t: Tree, node_index: NodeIndex) -> int:
    """Calculate how the node is center embedded."""
    node_depth: int = len(node_index)

    not_right_end: bool = False
    not_left_end: bool = False
    center_embedding_count: int = 0

    # Iterate over the path nodes from bottom.
    for i in range(node_depth):
        cur_node_index: NodeIndex = node_index[: node_depth - i]
        cur_parent: Tree = t[cur_node_index[:-1]]

        num_siblings: int = len(cur_parent)

        # Check if the current node is not left end.
        if (not not_left_end) and cur_node_index[-1] > 0:
            not_left_end = True

        # Check if the current node is not right end.
        if (not not_right_end) and cur_node_index[-1] < num_siblings - 1:
            not_right_end = True

        # The current node is center embedded.
        if not_left_end and not_right_end:
            center_embedding_count += 1
            not_left_end = False
            not_right_end = False

    return center_embedding_count


def max_center_embedding(t: Tree, node_index: NodeIndex) -> float:
    """Calculate the maximum degree of center embedding of the phrase leaves."""
    phrase: Tree = t[node_index]

    num_leaves: int = len(phrase.leaves())

    cur_max: int = 0
    # Iterate over the leaves.
    for i in range(num_leaves):
        leaf_index: NodeIndex = phrase.leaf_treeposition(i)

        tmp_center_embedding: int = center_embedding_outside(phrase, leaf_index)

        if tmp_center_embedding > cur_max:
            cur_max = tmp_center_embedding

    return cur_max


def normalized_max_center_embedding(t: Tree, node_index: NodeIndex) -> float:
    """Calculate the maximum degree of center embedding of the phrase leaves."""
    phrase: Tree = t[node_index]

    num_leaves: int = len(phrase.leaves())

    if num_leaves <= 2:
        return 0.0
    else:
        max_center_emb: float = max_center_embedding(t=t, node_index=node_index)

        center_emb_limit: int = np.floor((num_leaves - 1) / 2)

        normalized_center_embedding: float = max_center_emb / center_emb_limit

        return normalized_center_embedding


def phrase_max_center_embedding(t: Tree, node_index: NodeIndex) -> float:
    """Calculate the maximum degree of center embedding of the phrase."""
    phrase: Tree = t[node_index]

    num_leaves: int = len(phrase.leaves())

    cur_max: int = 0
    # Iterate over the leaves.
    for i in range(num_leaves):
        leaf_index: NodeIndex = phrase.leaf_treeposition(i)

        leaf_parent_index: NodeIndex = leaf_index[:-1]

        tmp_center_embedding: int = center_embedding_outside(phrase, leaf_parent_index)

        if tmp_center_embedding > cur_max:
            cur_max = tmp_center_embedding

    return cur_max


def phrase_normalized_max_center_embedding(t: Tree, node_index: NodeIndex) -> float:
    """Calculate the maximum degree of center embedding of the phrases."""
    phrase: Tree = t[node_index]

    num_leaves: int = len(phrase.leaves())

    if num_leaves <= 3:
        return 0.0
    else:
        max_center_emb: float = phrase_max_center_embedding(t=t, node_index=node_index)

        center_emb_limit: int = np.ceil((num_leaves - 3) / 2)

        normalized_center_embedding: float = max_center_emb / center_emb_limit

        return normalized_center_embedding


##########################
## Left/Right Branching ##
##########################


def round_to_infinite(x: float) -> int:
    """Round a given value so that it has a larger abs."""
    return np.sign(x) * np.ceil(np.abs(x))


def calc_children_weights(num_children: int) -> list[float]:
    """Calculate the weights for children.

    If num_children=5, the weights are (-1, -1/2, 0, 1/2, 1).
    The weight is 0 for unary node.
    When a node is a leaf, the weights are not defined.
    """
    if num_children <= 1:
        return [0.0 for _ in range(num_children)]
    else:
        step: float = 1 / (np.floor(num_children / 2))

        weights: list[float] = []
        for i in range(num_children):
            # Calculate the index of children by setting the center as 0.
            # If there are 5 children, then (-2, -1, 0, 1, 2)
            # If there are 4, then (-2, -1, 1, 2)
            index_from_center: int = round_to_infinite(i - ((num_children - 1) / 2))
            weight: float = index_from_center * step

            weights.append(weight)

        return weights


def weighted_relative_difference(t: Tree) -> float:
    """Calculate the relative difference in the number of leaves of the direct children of the root.

    Although the difference is not defined for leaf nodes, this function returns 0.0 for simplicity.
    """

    weights: list[float] = calc_children_weights(len(t))
    child_num_leaves: list[int] = [get_num_leaves(child) for child in t] if not isinstance(t, Leaf) else []

    return np.sum([w * n for w, n in zip(weights, child_num_leaves)])


def shallow_relative_difference(t: Tree, node_index: NodeIndex) -> float:
    """This does not consider deeper descendants."""
    if isinstance(t, Leaf):
        return 0.0
    else:
        phrase: Tree = t[node_index]

        rel_diff: float = weighted_relative_difference(phrase)
        num_leaves: int = len(phrase.leaves())

        # Corner case
        # Return 0 since trees must be balanced when there are only 1 or 2 leaves.
        if num_leaves == 1 or num_leaves == 2:
            return 0.0

        normalized_rel_diff: float = rel_diff / (num_leaves - 2)

        return normalized_rel_diff


def relative_corrected_colles_index(t: Tree, node_index: NodeIndex) -> float:
    if isinstance(t, Leaf):
        return 0.0
    else:
        phrase: Tree = t[node_index]

        # First, calculate (unnomalized) Colles index.
        nr_colles_index: float = np.sum([weighted_relative_difference(v) for v in iterate_over_inner_nodes(phrase)])

        num_leaves: int = len(phrase.leaves())

        # Corner case
        # Return 0 since trees must be balanced when there are only 1 or 2 leaves.
        if num_leaves == 1 or num_leaves == 2:
            return 0.0

        nr_corrected_colles_index: float = (2 * nr_colles_index) / ((num_leaves - 1) * (num_leaves - 2))

        return nr_corrected_colles_index


def relative_equal_weights_corrected_colles_index(t: Tree, node_index: NodeIndex) -> float:
    if isinstance(t, Leaf):
        return 0.0
    else:
        phrase: Tree = t[node_index]

        sum_normalized_diff: float = 0
        for v in iterate_over_inner_nodes(phrase):
            num_leaves_v: int = len(v.leaves())

            if num_leaves_v > 2:
                sum_normalized_diff += weighted_relative_difference(v) / (num_leaves_v - 2)

        num_leaves: int = len(phrase.leaves())

        # Corner case
        # Return 0 since trees must be balanced when there are only 1 or 2 leaves.
        if num_leaves == 1 or num_leaves == 2:
            return 0.0

        nr_equal_weights_colles_index: float = sum_normalized_diff / (num_leaves - 2)

        return nr_equal_weights_colles_index


def relative_rogers_j_index(t: Tree, node_index: NodeIndex) -> float:
    if isinstance(t, Leaf):
        return 0.0
    else:
        phrase: Tree = t[node_index]

        # First, calculate (unnomalized) Colles index.
        # sum_sign: float = np.sum([np.sign(weighted_relative_difference(v)) for v in iterate_over_inner_nodes(phrase)])

        # Ad hoc: but this is necessary for numerical stability.
        PREC: float = 10e-6
        weighted_rel_diff_list: list[float] = []
        for v in iterate_over_inner_nodes(phrase):
            wrd = weighted_relative_difference(v)
            if abs(wrd) < PREC:
                weighted_rel_diff_list.append(0.0)
            else:
                weighted_rel_diff_list.append(wrd)

        sum_sign: float = np.sum([np.sign(wrd) for wrd in weighted_rel_diff_list])

        num_leaves: int = len(phrase.leaves())

        # Corner case
        # Return 0 since trees must be balanced when there are only 1 or 2 leaves.
        if num_leaves == 1 or num_leaves == 2:
            return 0.0

        nr_staircaseness: float = sum_sign / (num_leaves - 2)

        return nr_staircaseness


##########
## Test ##
##########
if __name__ == "__main__":
    # T: dummy tag
    # C: dummy category
    t: Tree = nltk.Tree.fromstring("(C (C (C T (C (C T (C (C T T) T)) (C T (C T T)) (C T)) T)) T T)")

    tree1_index: NodeIndex = (0, 0, 1, 0)
    tree2_index: NodeIndex = (0, 0, 1)

    # Leaf size
    assert num_leaves_inside(t, tree1_index) == 4
    assert num_leaves_inside(t, tree2_index) == 8

    assert num_leaves_outside_left(t, tree1_index) == 1
    assert num_leaves_outside_left(t, tree2_index) == 1

    assert num_leaves_outside_right(t, tree1_index) == 7
    assert num_leaves_outside_right(t, tree2_index) == 3

    assert num_leaves_outside(t, tree1_index) == 8
    assert num_leaves_outside(t, tree2_index) == 4

    assert span_ratio(t, tree1_index) == (4 / 12)
    assert span_ratio(t, tree2_index) == (8 / 12)

    # Tree height
    assert node_depth(t, tree1_index) == 4
    assert node_depth(t, tree2_index) == 3

    # Flatness
    assert aspect_ratio(t, tree1_index) == (3 / 4)
    assert aspect_ratio(t, tree2_index) == (7 / 8)

    assert flatness(t, tree1_index) == (4 / 3)
    assert flatness(t, tree2_index) == (8 / 7)

    # Center embedding
    assert center_embedding_outside(t, tree1_index) == 1
    assert center_embedding_outside(t, tree2_index) == 1

    assert max_center_embedding(t, tree1_index) == 1
    assert max_center_embedding(t, tree2_index) == 2

    # completely center embedding tree
    ct: Tree = nltk.Tree.fromstring("(C T (C (C T (C (C T (C (C T (C (C T T) T)) T)) T)) T))")
    ct2: Tree = nltk.Tree.fromstring("(C T (C T (C T (C T (C T (C T (C T T T) T) T) T) T) T) T)")
    tree3_index: NodeIndex = (1, 0, 1, 0)
    tree4_index: NodeIndex = (1, 1, 1)

    assert center_embedding_outside(ct, tree3_index) == 2
    assert center_embedding_outside(ct2, tree4_index) == 3

    assert max_center_embedding(ct, tree3_index) == 2
    assert max_center_embedding(ct2, tree4_index) == 4

    # Left/Right branching
    assert shallow_relative_difference(t, tree1_index) == ((3 - 1) / ((4 - 1) - 1))
    assert shallow_relative_difference(t, tree2_index) == ((1 * 1 + 0 * 3 - 1 * 4) / (8 - 1 - 1))

    assert relative_corrected_colles_index(t, tree1_index) == 2 / 6
    assert relative_corrected_colles_index(t, tree2_index) == -1 / 21

    assert relative_equal_weights_corrected_colles_index(t, tree1_index) == 0.0 / 4
    assert relative_equal_weights_corrected_colles_index(t, tree2_index) == 1 / 12

    assert relative_rogers_j_index(t, tree1_index) == 0.0 / 3
    assert relative_rogers_j_index(t, tree2_index) == 0.0

    # Check for flipping.
    tree1_index_flip: NodeIndex = (2, 0, 1, 2)
    tree2_index_flip: NodeIndex = (2, 0, 1)
    assert shallow_relative_difference(t, tree1_index) == -shallow_relative_difference(flip_tree(t), tree1_index_flip)
    assert shallow_relative_difference(t, tree2_index) == -shallow_relative_difference(flip_tree(t), tree2_index_flip)

    assert relative_corrected_colles_index(t, tree1_index) == -relative_corrected_colles_index(
        flip_tree(t), tree1_index_flip
    )
    assert relative_corrected_colles_index(t, tree2_index) == -relative_corrected_colles_index(
        flip_tree(t), tree2_index_flip
    )

    assert relative_equal_weights_corrected_colles_index(
        t, tree1_index
    ) == -relative_equal_weights_corrected_colles_index(flip_tree(t), tree1_index_flip)
    assert relative_equal_weights_corrected_colles_index(
        t, tree2_index
    ) == -relative_equal_weights_corrected_colles_index(flip_tree(t), tree2_index_flip)

    assert relative_rogers_j_index(t, tree1_index) == -relative_rogers_j_index(flip_tree(t), tree1_index_flip)
    assert relative_rogers_j_index(t, tree2_index) == -relative_rogers_j_index(flip_tree(t), tree2_index_flip)
