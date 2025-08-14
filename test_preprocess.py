import sys
import unittest

import nltk
from get_treebank import collapse_unary
from tree_shape_measure import (
    NodeIndex,
    fixed_aspect_ratio,
    phrase_normalized_max_center_embedding,
    relative_corrected_colles_index,
    relative_equal_weights_corrected_colles_index,
    relative_rogers_j_index,
)


class TestModels(unittest.TestCase):
    def test_tree_shape_measure(self):
        right_branching = "(X a (X a (X a (X a (X a a)))))"
        left_branching = "(X (X (X (X (X a a) a) a) a) a)"
        flat = "(X a a a a a a)"
        center_emb = "(X a (X a (X a b) b) b)"
        center_emb_bin_left = "(X (X a (X (X a (X a b)) b)) b)"
        center_emb_bin_right = "(X a (X (X a (X (X a b) b)) b))"

        right_branching_t: nltk.Tree = nltk.Tree.fromstring(right_branching)
        left_branching_t: nltk.Tree = nltk.Tree.fromstring(left_branching)
        flat_t: nltk.Tree = nltk.Tree.fromstring(flat)
        center_emb_t: nltk.Tree = nltk.Tree.fromstring(center_emb)
        center_emb_bin_left_t: nltk.Tree = nltk.Tree.fromstring(center_emb_bin_left)
        center_emb_bin_right_t: nltk.Tree = nltk.Tree.fromstring(center_emb_bin_right)

        root_node_index: NodeIndex = ()

        self.assertEqual(fixed_aspect_ratio(t=right_branching_t, node_index=root_node_index), 1)
        self.assertEqual(fixed_aspect_ratio(t=left_branching_t, node_index=root_node_index), 1)
        self.assertEqual(fixed_aspect_ratio(t=flat_t, node_index=root_node_index), 0)
        self.assertEqual(fixed_aspect_ratio(t=center_emb_t, node_index=root_node_index), 2 / 4)

        self.assertEqual(phrase_normalized_max_center_embedding(t=right_branching_t, node_index=root_node_index), 0)
        self.assertEqual(phrase_normalized_max_center_embedding(t=left_branching_t, node_index=root_node_index), 0)
        self.assertEqual(phrase_normalized_max_center_embedding(t=flat_t, node_index=root_node_index), 0)
        self.assertEqual(phrase_normalized_max_center_embedding(t=center_emb_t, node_index=root_node_index), 1)
        self.assertEqual(phrase_normalized_max_center_embedding(t=center_emb_bin_left_t, node_index=root_node_index), 1)
        self.assertEqual(
            phrase_normalized_max_center_embedding(t=center_emb_bin_right_t, node_index=root_node_index), 1
        )

        self.assertAlmostEqual(relative_corrected_colles_index(t=right_branching_t, node_index=root_node_index), 1)
        self.assertAlmostEqual(relative_corrected_colles_index(t=left_branching_t, node_index=root_node_index), -1)
        self.assertAlmostEqual(relative_corrected_colles_index(t=flat_t, node_index=root_node_index), 0)
        self.assertAlmostEqual(relative_corrected_colles_index(t=center_emb_t, node_index=root_node_index), 0)

        self.assertAlmostEqual(
            relative_equal_weights_corrected_colles_index(t=right_branching_t, node_index=root_node_index), 1
        )
        self.assertAlmostEqual(
            relative_equal_weights_corrected_colles_index(t=left_branching_t, node_index=root_node_index), -1
        )
        self.assertAlmostEqual(relative_equal_weights_corrected_colles_index(t=flat_t, node_index=root_node_index), 0)
        self.assertAlmostEqual(
            relative_equal_weights_corrected_colles_index(t=center_emb_t, node_index=root_node_index), 0
        )

        self.assertAlmostEqual(relative_rogers_j_index(t=right_branching_t, node_index=root_node_index), 1)
        self.assertAlmostEqual(relative_rogers_j_index(t=left_branching_t, node_index=root_node_index), -1)
        self.assertAlmostEqual(relative_rogers_j_index(t=flat_t, node_index=root_node_index), 0)
        self.assertAlmostEqual(relative_rogers_j_index(t=center_emb_t, node_index=root_node_index), 0)

    def test_collapse_unary(self):
        tree_str = "(S t1 (X (Y t2 t3)) t4 (Z (W t5)) (P t6) (A (B t7) (C t8)))"
        unary_collapsed_tree_str = "(S t1 (X-Y t2 t3) t4 Z-W-t5 P-t6 (A B-t7 C-t8))"

        tree = nltk.Tree.fromstring(s=tree_str)
        collapsed = collapse_unary(tree)
        assert isinstance(collapsed, nltk.Tree)

        target_str = collapsed.pformat(margin=sys.maxsize)

        self.assertEqual(target_str, unary_collapsed_tree_str)


if __name__ == "__main__":
    unittest.main()
