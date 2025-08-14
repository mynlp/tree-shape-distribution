"""Convert treebank files into single ptb-style file."""

import glob
import re
import sys
from pathlib import Path

import myutils
import nltk
from mylogger import main_logger

logger = main_logger.getChild(__name__)

##########
## Util ##
##########


class DelexicalizationError(Exception):
    pass


def delexicalize_with_preterminal(t: nltk.Tree) -> nltk.Tree:
    """Use preterminals as leaves."""
    # assert t.height() > 2
    if not (t.height() > 2):
        raise DelexicalizationError

    new_childs = []
    for child in t:
        # assert isinstance(child, nltk.Tree)
        if not isinstance(child, nltk.Tree):
            raise DelexicalizationError

        # Since the tree is traversed in a top-down manner, preterminals are accessed before leaves.
        if child.height() == 2:
            # Sometimes, preterminals have more than one word somehow, but we simply replace them with their parent.
            # assert len(child) == 1

            # child is preterminal.
            preterminal: str = child.label()
            new_childs.append(preterminal)
        else:
            assert child.height() > 2
            # child is non-terminal but not preterminal.
            new_childs.append(delexicalize_with_preterminal(child))

    return nltk.Tree(t.label(), new_childs)


def delexicalize_with_prepreterminal(t: nltk.Tree) -> nltk.Tree:
    """Use prepreterminals as leaves."""
    assert t.height() > 3

    new_childs = []
    for child in t:
        assert isinstance(child, nltk.Tree)
        # Since the tree is traversed in a top-down manner, prepreterminals are accessed before preterminals.
        if child.height() == 3:
            # child is prepreterminal.

            # In some cases prepreterminals have more than one child.
            # This odd case happens on SPMRL Hebrew.
            prepreterminal: str = child.label()
            new_childs.append(prepreterminal)
        else:
            assert child.height() > 3
            # child is non-terminal but not preterminal.
            new_childs.append(delexicalize_with_prepreterminal(child))

    return nltk.Tree(t.label(), new_childs)


def min_leaf_depth(t: nltk.Tree) -> int:
    leaf_depths: list[int] = [len(t.leaf_treeposition(leaf_i)) for leaf_i in range(len(t.leaves()))]
    return min(leaf_depths)


def delexicalize_with_prepreterminal_spmrl_polish(t: nltk.Tree) -> nltk.Tree:
    """Use lowest nonterminals as leaves.

    This function is necessary to deal with nested prepreterminals in SPMRL Polish.
    """
    assert t.height() > 3

    new_childs = []
    for child in t:
        assert isinstance(child, nltk.Tree)
        # Since the tree is traversed in a top-down manner, prepreterminals are accessed before preterminals.
        # In SPMRL Polish, prepreterminals are sometimes nested; so, we need to calculate the min_leaf depth.
        if min_leaf_depth(child) == 2:
            # child is prepreterminal.

            # In some cases prepreterminals have more than one child.
            # This odd case happens on SPMRL Hebrew.
            prepreterminal: str = child.label()
            new_childs.append(prepreterminal)
        else:
            assert child.height() > 3
            # child is non-terminal but not preterminal.
            new_childs.append(delexicalize_with_prepreterminal_spmrl_polish(child))

    return nltk.Tree(t.label(), new_childs)


def collapse_unary(t: nltk.Tree) -> nltk.Tree | str:
    if len(t) == 1:
        # Current node is unary.

        # The only child.
        child = t[0]

        # Child is leaf
        if isinstance(child, str):
            # Concat current nonterminal with the leaf.
            new_leaf = f"{t.label()}-{child}"

            return new_leaf

        # Child is tree.
        else:
            assert isinstance(child, nltk.Tree)
            new_label = f"{t.label()}-{child.label()}"
            tmp_new_child = nltk.Tree(new_label, [c for c in child])

            return collapse_unary(tmp_new_child)

    else:
        # Currnet node is not uanry.
        new_childs = []
        for child in t:
            if isinstance(child, str):
                # Child is leaf.

                new_childs.append(child)
            else:
                # Child is Tree.
                assert isinstance(child, nltk.Tree)
                new_childs.append(collapse_unary(t=child))

        return nltk.Tree(t.label(), new_childs)


def remove_preterminals(t: nltk.Tree) -> nltk.Tree:
    new_childs = []
    for child in t:
        if isinstance(child, str):
            new_childs.append(child)

        # elif len(child) == 1 and child.height() == 2:
        elif child.height() == 2:
            assert len(child) == 1
            # child is preterminal.
            new_childs.append(child[0])

        else:
            # child is non-terminal but not preterminal.
            new_childs.append(remove_preterminals(child))

    return nltk.Tree(t.label(), new_childs)


def remove_prepreterminals(t: nltk.Tree) -> nltk.Tree:
    new_childs = []
    for child in t:
        if isinstance(child, str):
            new_childs.append(child)

        elif child.height() <= 3:
            # child is prepreterminal.

            # Since the tree is traversed in a top-down manner, prepreterminals are accessed before preterminals.
            assert child.height() != 2

            # This can deal with some cases where prepreterminals have more than one child.
            # This odd case happens on SPMRL Hebrew.
            new_childs.extend(child.leaves())

        else:
            # child is non-terminal but not preterminal.
            new_childs.append(remove_prepreterminals(child))

    return nltk.Tree(t.label(), new_childs)


def check_null_element(t: nltk.Tree, treebank_type: str) -> bool:
    match treebank_type:
        case "ptb" | "ctb":
            return t.label() == "-NONE-"

        case "ftb":
            if len(t.leaves()) == 1:
                text = t.leaves()[0]
                if text == "*T*":
                    return True

            return False

        case "npcmj" | "kortb":
            if len(t.leaves()) == 1:
                text = t.leaves()[0]

                # null elements without indexing
                if text.startswith("*") and text.endswith("*"):
                    return True
                # null elements with indexing
                if re.match("^\*(.*)\*-[0-9]+$", text):
                    return True

            return False
        case _:
            raise Exception(f"No such treebank_type: {treebank_type}")


def remove_null_element_sub(t: nltk.Tree | str, treebank_type: str) -> str:
    if isinstance(t, str):
        return t
    elif check_null_element(t, treebank_type):
        # Drop null element.
        return ""
    else:
        subtree_str_l: list[str] = []

        for child in t:
            subtree_str: str = remove_null_element_sub(child, treebank_type=treebank_type)
            subtree_str_l.append(subtree_str)

        children_str: str = " ".join(subtree_str_l)

        # Check if all the children are null elements.
        # If all children are null, simply remove the parent.
        if children_str.strip() == "":
            return ""
        else:
            tree_str: str = f"({t.label()} {children_str})"

            return tree_str


class ReturnEmptyError(Exception):
    pass


def remove_null_element(t: nltk.Tree, treebank_type: str) -> nltk.Tree:
    s = remove_null_element_sub(t, treebank_type=treebank_type)
    if s == "":
        # NPCMJ has a tree that consists of only null elements...
        raise ReturnEmptyError
    else:
        return nltk.Tree.fromstring(s)


def normalize_category(cat_str: str) -> str:
    if cat_str.startswith("-") and cat_str.endswith("-"):
        # Leave categories, e.g., -LRB-
        return cat_str

    # Get the left side of '-'
    cat: str = cat_str.split("-")[0]

    # Get the left side of '='
    cat: str = cat.split("=")[0]

    # Get the left side of '|'
    cat: str = cat.split("|")[0]

    # Get the left side of ';'
    # Spcific to Keyaki Treebank.
    cat: str = cat.split(";")[0]

    # Get the left side of '{'
    # Spcific to Keyaki Treebank.
    cat: str = cat.split("{")[0]

    # Specific to SPMRL
    cat: str = cat.split("##")[0]

    return cat


def normalize_cat_tree(t: nltk.Tree) -> None:
    cur_label: str = t.label()
    # Set normalized label.
    t.set_label(label=normalize_category(cur_label))

    for child in t:
        if isinstance(child, str):
            # Do nothing.
            continue
        else:
            # Otherwise, the child is nltk.Tree.
            normalize_cat_tree(child)

    return


############################
## Treebank Load Function ##
############################


def get_ptb_data(dirpath: Path, output_filepath: Path, min_num_leaves: int):
    # Use sections 2-21 (train section).
    # Pad 0.
    sections: list[str] = [f"{n:0=2}" for n in range(2, 21)]

    t_str_l: list[str] = []

    for section in sections:
        file_count = 0

        for file_path in glob.glob(str(dirpath.joinpath(section, "*"))):
            file_count += 1

            with Path(file_path).open(mode="r") as f:
                cur_str: str = ""
                num_left_bracket: int = 0
                num_right_bracket: int = 0

                for l in f:
                    if l == "\n":
                        continue
                    else:
                        cur_str += l
                        num_left_bracket += l.count("(")
                        num_right_bracket += l.count(")")

                        if num_left_bracket == num_right_bracket:
                            t = nltk.Tree.fromstring(cur_str, remove_empty_top_bracketing=True)

                            # Remove null element.
                            t = remove_null_element(t, "ptb")

                            # Normailze category.
                            normalize_cat_tree(t)

                            # Remove preterminals.
                            # t = remove_preterminals(t=t)

                            # Delexicalize (instead of removing preterminals).
                            t = delexicalize_with_preterminal(t=t)

                            # This may return non-tree when a tree contains only unary nonterminals.
                            t = collapse_unary(t=t)

                            # Check if t is tree and only collect trees that have more leaves than min_num_leaves.
                            if isinstance(t, nltk.Tree) and len(t.leaves()) >= min_num_leaves:
                                t_str: str = t.pformat(margin=sys.maxsize)

                                t_str_l.append(t_str)

                            cur_str = ""
                            num_left_bracket = 0
                            num_right_bracket = 0
        if file_count <= 0:
            raise Exception(f"No file found in section {section}")

    with output_filepath.open(mode="w") as g:
        g.write("\n".join(t_str_l))


def get_npcmj_data(dirpath: Path, output_filepath: Path, min_num_leaves: int):
    t_str_l: list[str] = []

    # Use all files.
    for file_path in glob.glob(str(dirpath.joinpath("*"))):
        with Path(file_path).open(mode="r") as f:
            cur_str: str = ""
            num_left_bracket: int = 0
            num_right_bracket: int = 0

            for l in f:
                if l == "\n":
                    continue
                else:
                    cur_str += l
                    num_left_bracket += l.count("(")
                    num_right_bracket += l.count(")")

                    if num_left_bracket == num_right_bracket:
                        id_t = nltk.Tree.fromstring(cur_str, remove_empty_top_bracketing=True)

                        # Do not include id subtree.
                        try:
                            assert len(id_t) == 2
                            assert id_t[1].label() == "ID"
                        except:
                            logger.info(f"{file_path}")
                            logger.info(f"Format error: {id_t}")
                            logger.info("Skip the sample.")

                            # Just discard invalid data.
                            # We only found one such sample though.
                            cur_str = ""
                            num_left_bracket = 0
                            num_right_bracket = 0
                            continue
                            # exit(1)

                        t = id_t[0]

                        # Remove null element.
                        try:
                            t = remove_null_element(t, "npcmj")
                        except ReturnEmptyError:
                            logger.info(f"{file_path}")
                            logger.info(f"Null tree: {id_t}")
                            logger.info("Skip the sample.")

                            # Just discard invalid data.
                            cur_str = ""
                            num_left_bracket = 0
                            num_right_bracket = 0
                            continue

                        # Normailze category.
                        normalize_cat_tree(t)

                        ## Remove preterminals.
                        # t = remove_preterminals(t=t)

                        # Delexicalize (instead of removing preterminals).
                        # t = delexicalize_with_preterminal(t=t)
                        try:
                            t = delexicalize_with_preterminal(t=t)
                        except DelexicalizationError:
                            logger.info(f"{file_path}")
                            logger.info(f"Delexicalization: {id_t}")
                            logger.info("Skip the sample.")

                            # Just discard invalid data.
                            cur_str = ""
                            num_left_bracket = 0
                            num_right_bracket = 0
                            continue

                        # This may return non-tree when a tree contains only unary nonterminals.
                        t = collapse_unary(t=t)

                        # Check if t is tree and only collect trees that have more leaves than min_num_leaves.
                        if isinstance(t, nltk.Tree) and len(t.leaves()) >= min_num_leaves:
                            t_str: str = t.pformat(margin=sys.maxsize)

                            t_str_l.append(t_str)

                        cur_str = ""
                        num_left_bracket = 0
                        num_right_bracket = 0

    with output_filepath.open(mode="w") as g:
        g.write("\n".join(t_str_l))


def get_ctb_data(dirpath: Path, output_filepath: Path, min_num_leaves: int):
    t_str_l: list[str] = []

    # Set sections.
    train_sections = list(range(1, 270 + 1)) + list(range(440, 1151 + 1))
    # dev_sections = list(range(301, 325 + 1))
    # test_sections = list(range(271, 300 + 1))

    # Use train sections.
    for section in train_sections:
        section_file = dirpath.joinpath(f"chtb_{section:0=3}.fid")

        # Simply skip if there is not corresponding file for the fid.
        if not section_file.exists():
            logger.info(f"Skipt fid {section}; there is no corresponding file.")
            continue

        with section_file.open(mode="r", encoding="GB2312") as f:
            cur_str: str = ""
            num_left_bracket: int = 0
            num_right_bracket: int = 0

            for l in f:
                if len(l.lstrip()) == 0:
                    continue
                elif l.lstrip()[0] == "<":
                    continue
                else:
                    cur_str += l
                    num_left_bracket += l.count("(")
                    num_right_bracket += l.count(")")

                    if num_left_bracket == num_right_bracket:
                        t = nltk.Tree.fromstring(cur_str, remove_empty_top_bracketing=True)

                        # Remove null element.
                        t = remove_null_element(t, "ctb")

                        # Normailze category.
                        normalize_cat_tree(t)

                        ## Remove preterminals.
                        # t = remove_preterminals(t=t)

                        # Delexicalize (instead of removing preterminals).
                        try:
                            t = delexicalize_with_preterminal(t=t)
                        except:
                            logger.info(f"{t}")

                        # This may return non-tree when a tree contains only unary nonterminals.
                        t = collapse_unary(t=t)

                        # Check if t is tree and only collect trees that have more leaves than min_num_leaves.
                        if isinstance(t, nltk.Tree) and len(t.leaves()) >= min_num_leaves:
                            t_str: str = t.pformat(margin=sys.maxsize)

                            t_str_l.append(t_str)

                        cur_str = ""
                        num_left_bracket = 0
                        num_right_bracket = 0

    with output_filepath.open(mode="w") as g:
        g.write("\n".join(t_str_l))


def get_kortb_data(dirpath: Path, output_filepath: Path, min_num_leaves: int):
    t_str_l: list[str] = []

    # TODO: only use train sections??
    # Use all files.
    for file_path in glob.glob(str(dirpath.joinpath("*"))):
        # The Files are in EUC-KR encoding.
        with Path(file_path).open(mode="r", encoding="EUC-KR") as f:
            cur_str: str = ""
            num_left_bracket: int = 0
            num_right_bracket: int = 0

            for l_euckr in f:
                # Convert to utf-8.
                l = l_euckr.encode("utf-8").decode("utf-8")

                if l == "\n" or l.startswith(";;"):
                    continue
                else:
                    cur_str += l
                    num_left_bracket += l.count("(")
                    num_right_bracket += l.count(")")

                    if num_left_bracket == num_right_bracket:
                        # But, kortb has no top bracket.
                        t = nltk.Tree.fromstring(cur_str, remove_empty_top_bracketing=True)

                        # Remove null element.
                        t = remove_null_element(t, "kortb")

                        # Normailze category.
                        normalize_cat_tree(t)

                        ## Remove preterminals.
                        # t = remove_preterminals(t=t)

                        # Delexicalize (instead of removing preterminals).
                        t = delexicalize_with_preterminal(t=t)

                        # This may return non-tree when a tree contains only unary nonterminals.
                        t = collapse_unary(t=t)

                        # Check if t is tree and only collect trees that have more leaves than min_num_leaves.
                        if isinstance(t, nltk.Tree) and len(t.leaves()) >= min_num_leaves:
                            t_str: str = t.pformat(margin=sys.maxsize)

                            t_str_l.append(t_str)

                        cur_str = ""
                        num_left_bracket = 0
                        num_right_bracket = 0

    with output_filepath.open(mode="w") as g:
        g.write("\n".join(t_str_l))


def get_ftb_data(dirpath: Path, output_filepath: Path, min_num_leaves: int):
    t_str_l: list[str] = []

    # Use all files.
    for file_path in glob.glob(str(dirpath.joinpath("*"))):
        # The Files are in ISO-8859-1 encoding.
        with Path(file_path).open(mode="r", encoding="ISO-8859-1") as f:
            # One tree per line.
            for l_encoded in f:
                # Convert to utf-8.
                l = l_encoded.encode("utf-8").decode("utf-8")

                # Each line starts with "-id\t" followed by ptb-style tree string.
                if l.startswith("-"):
                    # Remove the id.
                    cur_str = " ".join(l.split()[1:])

                    t = nltk.Tree.fromstring(cur_str, remove_empty_top_bracketing=True)

                    # Remove null element.
                    t = remove_null_element(t, "ftb")

                    # Normailze category.
                    normalize_cat_tree(t)

                    ## Remove preterminals.
                    # t = remove_preterminals(t=t)

                    # Delexicalize (instead of removing preterminals).
                    t = delexicalize_with_preterminal(t=t)

                    # This may return non-tree when a tree contains only unary nonterminals.
                    t = collapse_unary(t=t)

                    # Check if t is tree and only collect trees that have more leaves than min_num_leaves.
                    if isinstance(t, nltk.Tree) and len(t.leaves()) >= min_num_leaves:
                        t_str: str = t.pformat(margin=sys.maxsize)

                        t_str_l.append(t_str)

    with output_filepath.open(mode="w") as g:
        g.write("\n".join(t_str_l))


def get_spmrl_data(dirpath: Path, output_filepath: Path, lang: str, min_num_leaves: int):
    """
    The trees in SPMRL dataset do not contain null elements.
    Args:
        lang: string key specifying the target language: Basque, French, German, Hebrew, Hungarian, Korean, Polish, swedish. Note that for Swedish somehow the initial char is not capitalized.
    """

    # Check lang key.
    match lang:
        case "Basque" | "French" | "German" | "Hebrew" | "Hungarian" | "Korean" | "Polish" | "swedish":
            pass
        case "Swedish":
            raise Exception(f"Invalid lang key: {lang}. Maybe 'swedish'...?")
        case _:
            raise Exception(f"Invalid lang key: {lang}")

    t_str_l: list[str] = []

    # Use only train split.
    match lang:
        case "Basque" | "French" | "German" | "Hungarian" | "Korean" | "Polish":
            file_paths = [
                dirpath.joinpath("train", f"train.{lang}.gold.ptb"),
                # dirpath.joinpath("dev", f"dev.{lang}.gold.ptb"),
                # dirpath.joinpath("test", f"test.{lang}.gold.ptb"),
            ]
        case "Hebrew" | "swedish":
            # Since these two languages have only small amount of data, we use the train5k split.
            file_paths = [
                dirpath.joinpath("train5k", f"train5k.{lang}.gold.ptb"),
                # dirpath.joinpath("dev", f"dev.{lang}.gold.ptb"),
                # dirpath.joinpath("test", f"test.{lang}.gold.ptb"),
            ]

    for file_path in file_paths:
        # The Files are in utf-8.
        with file_path.open(mode="r", encoding="utf-8") as f:
            # One tree per line.
            for l in f:
                if l == "\n":
                    continue
                else:
                    cur_str = l

                    # But, kortb has no top bracket.
                    t = nltk.Tree.fromstring(cur_str, remove_empty_top_bracketing=True)

                    # Process top brackets.
                    match lang:
                        # Those with non-empty top bracket
                        case "Basque" | "Hebrew" | "Korean":
                            try:
                                assert len(t) == 1
                            except:
                                logger.info(f"{file_path}")
                                logger.info(f"Error: {t}")
                                logger.info("Skip the sample.")
                                continue

                            # Remove the top bracket.
                            t = t[0]

                        # Those with empty top bracket
                        case "French" | "swedish":
                            # Do nothing because the empty top brackets are already removed.
                            pass
                        # Those with no top bracket
                        case "German" | "Polish" | "Hungarian":
                            # Need not do anything.
                            pass

                    # Null elements are already removed.

                    # Normailze category.
                    normalize_cat_tree(t)

                    # Remove preterminals.
                    match lang:
                        case "Hebrew":
                            # SPMRL Hebrew has two layer preterminals.
                            # But, note that in some cases pre-preterminals are not not unary (though the reason is not clear)
                            # t = remove_prepreterminals(t=t)

                            # Delexicalize (instead of removing preterminals).
                            t = delexicalize_with_prepreterminal(t=t)

                        case "Polish":
                            # SPMRL Polish has two layer preterminals.
                            # But, note that in some cases pre-preterminals are not not unary and nested
                            # t = remove_prepreterminals(t=t)

                            # Delexicalize (instead of removing preterminals).
                            t = delexicalize_with_prepreterminal_spmrl_polish(t=t)

                        case _:
                            # t = remove_preterminals(t=t)

                            # Delexicalize (instead of removing preterminals).
                            # t = delexicalize_with_preterminal(t=t)
                            try:
                                t = delexicalize_with_preterminal(t=t)
                            except DelexicalizationError:
                                logger.info(f"{file_path}")
                                logger.info(f"DelexicalizationError: {t}")
                                logger.info("Skip the sample.")
                                continue

                    # This may return non-tree when a tree contains only unary nonterminals.
                    t = collapse_unary(t=t)

                    # Check if t is tree and only collect trees that have more leaves than min_num_leaves.
                    if isinstance(t, nltk.Tree) and len(t.leaves()) >= min_num_leaves:
                        t_str: str = t.pformat(margin=sys.maxsize)

                        t_str_l.append(t_str)

    with output_filepath.open(mode="w") as g:
        g.write("\n".join(t_str_l))


def get_debug_data(dirpath: Path, output_filepath: Path, min_num_leaves: int):
    t_str_l: list[str] = []

    for file_path in glob.glob(str(dirpath.joinpath("*"))):
        with Path(file_path).open(mode="r") as f:
            cur_str: str = ""
            num_left_bracket: int = 0
            num_right_bracket: int = 0

            for l in f:
                if l == "\n":
                    continue
                else:
                    cur_str += l
                    num_left_bracket += l.count("(")
                    num_right_bracket += l.count(")")

                    if num_left_bracket == num_right_bracket:
                        t = nltk.Tree.fromstring(cur_str, remove_empty_top_bracketing=True)

                        # Remove null element.
                        t = remove_null_element(t, "ptb")

                        # Normailze category.
                        normalize_cat_tree(t)

                        ## Remove preterminals.
                        # t = remove_preterminals(t=t)

                        # Delexicalize (instead of removing preterminals).
                        t = delexicalize_with_preterminal(t=t)

                        # This may return non-tree when a tree contains only unary nonterminals.
                        t = collapse_unary(t=t)

                        # Check if t is tree and only collect trees that have more leaves than min_num_leaves.
                        if isinstance(t, nltk.Tree) and len(t.leaves()) >= min_num_leaves:
                            t_str: str = t.pformat(margin=sys.maxsize)

                            t_str_l.append(t_str)

                        cur_str = ""
                        num_left_bracket = 0
                        num_right_bracket = 0

    with output_filepath.open(mode="w") as g:
        g.write("\n".join(t_str_l))


class Args(myutils.BaseArgs):
    treebank_key: str
    output_filepath: str
    source_data_dir: str

    min_num_leaves: int

    # Null elements are removed by default.
    # Categories are normalized by default.
    # Unary nodes are concatenated and collapsed.


if __name__ == "__main__":
    # Set args.
    args = Args.parse_args()

    logger.info(f"Start loading {args.treebank_key} !!!!")

    save_filepath: Path = Path(args.output_filepath)
    save_filepath.parent.mkdir(parents=True, exist_ok=True)

    match args.treebank_key:
        case "ptb":
            get_ptb_data(
                dirpath=Path(args.source_data_dir),
                output_filepath=save_filepath,
                min_num_leaves=args.min_num_leaves,
            )
        case "ctb":
            get_ctb_data(
                dirpath=Path(args.source_data_dir),
                output_filepath=save_filepath,
                min_num_leaves=args.min_num_leaves,
            )
        case "npcmj":
            get_npcmj_data(
                dirpath=Path(args.source_data_dir),
                output_filepath=save_filepath,
                min_num_leaves=args.min_num_leaves,
            )
        case "kortb":
            get_kortb_data(
                dirpath=Path(args.source_data_dir),
                output_filepath=save_filepath,
                min_num_leaves=args.min_num_leaves,
            )
        case "ftb":
            get_ftb_data(
                dirpath=Path(args.source_data_dir),
                output_filepath=save_filepath,
                min_num_leaves=args.min_num_leaves,
            )

        case "spmrl_basque":
            get_spmrl_data(
                dirpath=Path(args.source_data_dir),
                output_filepath=save_filepath,
                lang="Basque",
                min_num_leaves=args.min_num_leaves,
            )

        case "spmrl_french":
            get_spmrl_data(
                dirpath=Path(args.source_data_dir),
                output_filepath=save_filepath,
                lang="French",
                min_num_leaves=args.min_num_leaves,
            )

        case "spmrl_german":
            get_spmrl_data(
                dirpath=Path(args.source_data_dir),
                output_filepath=save_filepath,
                lang="German",
                min_num_leaves=args.min_num_leaves,
            )

        case "spmrl_hebrew":
            get_spmrl_data(
                dirpath=Path(args.source_data_dir),
                output_filepath=save_filepath,
                lang="Hebrew",
                min_num_leaves=args.min_num_leaves,
            )

        case "spmrl_hungarian":
            get_spmrl_data(
                dirpath=Path(args.source_data_dir),
                output_filepath=save_filepath,
                lang="Hungarian",
                min_num_leaves=args.min_num_leaves,
            )

        case "spmrl_korean":
            get_spmrl_data(
                dirpath=Path(args.source_data_dir),
                output_filepath=save_filepath,
                lang="Korean",
                min_num_leaves=args.min_num_leaves,
            )

        case "spmrl_polish":
            get_spmrl_data(
                dirpath=Path(args.source_data_dir),
                output_filepath=save_filepath,
                lang="Polish",
                min_num_leaves=args.min_num_leaves,
            )

        case "spmrl_swedish":
            get_spmrl_data(
                dirpath=Path(args.source_data_dir),
                output_filepath=save_filepath,
                lang="swedish",
                min_num_leaves=args.min_num_leaves,
            )

        case "debug_data":  # PTB-style debug data.
            get_debug_data(
                dirpath=Path(args.source_data_dir),
                output_filepath=save_filepath,
                min_num_leaves=args.min_num_leaves,
            )

        case _:
            raise Exception(f"get_treebank not implemented for {args.treebank_key}")

    logger.info(f"Finish loading {args.treebank_key} !!!!")
