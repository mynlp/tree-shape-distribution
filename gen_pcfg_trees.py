import random
import sys
from pathlib import Path

import myutils
import nltk
from mylogger import main_logger
from tree_shape_measure import NodeIndex

logger = main_logger.getChild(__name__)

START_SYMBOL: str = "PCFG_START_SYMBOL"

###########
## Utils ##
###########


def get_tree_for_pcfg(t: nltk.Tree) -> nltk.Tree:
    all_parent_symbol: set[str] = set([sub_t.label() for sub_t in t.subtrees()])
    assert START_SYMBOL not in all_parent_symbol

    wrapped_tree = nltk.Tree(START_SYMBOL, [t])

    return wrapped_tree


def get_pcfg(tl: list[nltk.Tree]) -> nltk.grammar.PCFG:
    productions: list[nltk.grammar.Production] = []

    for t in tl:
        wrapped_tree = get_tree_for_pcfg(t=t)

        productions += wrapped_tree.productions()

    pcfg = nltk.induce_pcfg(start=nltk.grammar.Nonterminal(START_SYMBOL), productions=productions)

    return pcfg


class SentLenLimitError(Exception):
    pass


def sample_single_pcfg_tree(uniform: bool, pcfg: nltk.grammar.PCFG, sent_len_limit: int) -> nltk.Tree:
    """Breath-first PCFG tree sampling."""
    # Initial tree.
    start_nonterminal = nltk.grammar.Nonterminal(symbol=START_SYMBOL)
    tree = nltk.Tree(START_SYMBOL, [start_nonterminal])

    # Breath-first search.
    while True:
        # Calcel the process if it exceeds the sent len limit.
        # This is useful to avoid infinite loop.
        # Since unary productions are collapsed, counting both terminal and nonterminal leaves is enough.
        if len(tree.leaves()) > sent_len_limit:
            raise SentLenLimitError

        nonterminal_leaves: list[NodeIndex] = []
        for leaf_i in range(len(tree.leaves())):
            leaf_tree_pos: NodeIndex = tree.leaf_treeposition(leaf_i)

            if isinstance(tree[leaf_tree_pos], str):
                pass

            elif isinstance(tree[leaf_tree_pos], nltk.grammar.Nonterminal):
                nonterminal_leaves.append(leaf_tree_pos)

            else:
                raise Exception("something is wrong.")

        # Finish the process.
        if len(nonterminal_leaves) == 0:
            break

        # Otherwise, proceed the next step.
        for nonterminal_leaf_pos in nonterminal_leaves:
            # Get the nonterminal to substitute.
            nonterminal: nltk.grammar.Nonterminal = tree[nonterminal_leaf_pos]
            assert isinstance(nonterminal, nltk.grammar.Nonterminal)

            productions = pcfg.productions(lhs=nonterminal)
            rhs_candidates = [prod.rhs() for prod in productions]

            if uniform:
                # Force to sample from uniform distribution.
                # Setting weight 1.0 is enough (i.e., no need to normalize them).
                rule_probs = [1.0 for _ in productions]
            else:
                rule_probs = [prod.prob() for prod in productions]

            sampled_rhs = random.choices(population=rhs_candidates, weights=rule_probs, k=1)[0]

            tree[nonterminal_leaf_pos] = nltk.Tree(node=nonterminal.symbol(), children=list(sampled_rhs))

    # Ad hoc, but this is necessary to remove START_SYMBOL.
    sampled_tree = tree[0][0]
    assert isinstance(sampled_tree, nltk.Tree)
    assert sampled_tree.label() != START_SYMBOL

    return sampled_tree


def gen_pcfg_trees(
    output_filepath: Path,
    uniform: bool,
    pcfg: nltk.grammar.PCFG,
    num_data: int,
    sent_len_limit: int,
    min_num_leaves: int,
):
    t_str_l: list[str] = list()

    while len(t_str_l) < num_data:
        try:
            sampled_t = sample_single_pcfg_tree(uniform=uniform, pcfg=pcfg, sent_len_limit=sent_len_limit)

        except SentLenLimitError:
            continue

        # Format the nltk Tree to string.
        # Only collect trees that have more leaves than min_num_leaves.
        if len(sampled_t.leaves()) >= min_num_leaves:
            t_str: str = sampled_t.pformat(margin=sys.maxsize)

            t_str_l.append(t_str)

            # logger.info(f"{len(t_str_l)=}")

    with output_filepath.open(mode="w") as g:
        g.write("\n".join(t_str_l))


class Args(myutils.BaseArgs):
    base_dataset_filepath: str
    output_filepath: str

    uniform: bool

    num_data: int

    min_num_leaves: int

    random_seed: int


if __name__ == "__main__":
    # Set args.
    args = Args.parse_args()

    myutils.set_random_seed(random_seed=args.random_seed)

    base_trees: list[nltk.Tree] = []
    with Path(args.base_dataset_filepath).open(mode="r") as f:
        for l in f:
            t = nltk.Tree.fromstring(l)
            base_trees.append(t)

    logger.info("Start calculating MLE PCFG!!!!!!")
    pcfg = get_pcfg(tl=base_trees)
    logger.info("Finish calculating MLE PCFG!!!!!!")

    # debug
    logger.info(f"{pcfg.productions()=}")
    # debug

    # Set the same sent len limit as the base treebank.
    sent_len_limit: int = max([len(t.leaves()) for t in base_trees])
    logger.info(f"{sent_len_limit=}")

    logger.info("Start sampling pcfg trees!!!!!!")
    gen_pcfg_trees(
        output_filepath=Path(args.output_filepath),
        uniform=args.uniform,
        pcfg=pcfg,
        num_data=args.num_data,
        sent_len_limit=sent_len_limit,
        min_num_leaves=args.min_num_leaves,
    )
    logger.info("Finish sampling pcfg trees!!!!!!")
