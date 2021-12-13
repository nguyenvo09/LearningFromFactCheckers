from matchzoo.data_pack import DataPack
from .units import Vocabulary, VocabularyFC
from .build_unit_from_data_pack import build_unit_from_data_pack, build_unit_from_file
import json


def build_vocab_unit(
    data_pack: DataPack,
    mode: str = 'both',
    verbose: int = 1
) -> Vocabulary:
    """
    Build a :class:`preprocessor.units.Vocabulary` given `data_pack`.

    The `data_pack` should be preprocessed forehand, and each item in
    `text_left` and `text_right` columns of the `data_pack` should be a list
    of tokens.

    :param data_pack: The :class:`DataPack` to build vocabulary upon.
    :param mode: One of 'left', 'right', and 'both', to determine the source
    data for building the :class:`VocabularyUnit`.
    :param verbose: Verbosity.
    :return: A built vocabulary unit.

    """
    return build_unit_from_data_pack(
        unit=Vocabulary(),
        data_pack=data_pack,
        mode=mode,
        flatten=True, verbose=verbose
    )


def build_vocab_unit_from_file(
    vocab_file: str,
    verbose: int = 1
) -> Vocabulary:
    """
    Build a :class:`preprocessor.units.Vocabulary` given `vocab_file` in json format

    data for building the :class:`VocabularyUnit`.
    :param verbose: Verbosity.
    :return: A built vocabulary unit.

    """
    mapper = json.loads(open(vocab_file).read())
    return build_unit_from_file(
        unit=VocabularyFC(),
        terms=mapper,
        verbose=verbose
    )