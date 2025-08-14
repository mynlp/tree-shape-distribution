from dataclasses import dataclass


@dataclass
class WholeTreeResults:
    # Tree shape measures.

    # Metrics used in the paper.
    # Flatness.
    fixed_aspect_ratio: float

    # Non-linearity.
    phrase_normalized_max_center_emb: float

    # Branching Direction.
    colles: float
    equal_weights_colles: float
    rogers_j: float

    # Metrics not used in the paper.

    unary_collapsed_aspect_ratio: float
    aspect_ratio: float
    mean_degree: float

    max_center_emb: float
    normalized_max_center_emb: float
    phrase_max_center_emb: float
    height: int  # When the length of sentence is the same, if this value is smaller, then it is more closer to complete tree.

    num_leaves: int
