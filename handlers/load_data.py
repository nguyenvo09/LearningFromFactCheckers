import typing
import csv
from pathlib import Path

import pandas as pd
import time
import matchzoo
import os
from typing import List, Set, Tuple, Dict
from tqdm import tqdm
import torch
import numpy as np
import itertools
from handlers.output_handler import FileHandler
import torchvision
import torch_utils
from Models import pack
from setting_keywords import KeyWordSettings


def load_data2(
    data_root: str,
    stage: str = 'train',
    prefix: str = "Snopes"
) -> typing.Union[matchzoo.DataPack, tuple]:
    """
    Load WikiQA data.
    :param stage: One of `train`, `dev`, and `test`.
    :param task: Could be one of `ranking`, `classification` or a
        :class:`matchzoo.engine.BaseTask` instance.
    :param filtered: Whether remove the questions without correct answers.
    :param return_classes: `True` to return classes for classification task,
        `False` otherwise.
    :return: A DataPack unless `task` is `classificiation` and `return_classes`
        is `True`: a tuple of `(DataPack, classes)` in that case.
    """
    if stage not in ('train', 'dev', 'test', 'test2_hard', 'test3_hard'):
        raise ValueError("%s is not a valid stage. Must be one of `train`, `dev`, and `test`." % stage)

    # data_root = _download_data()
    data_root = data_root
    file_path = os.path.join(data_root, '%s.%s.tsv' % (prefix, stage))
    data_pack = _read_data2(file_path)
    return data_pack


def _read_data2(path):
    table = pd.read_csv(path, sep='\t', header=0, quoting=csv.QUOTE_NONE)

    def _prepare_decoder_input(text: str):
        return KeyWordSettings.start_token + " " + text

    def _prepare_decoder_target(text: str):
        return text + " " + KeyWordSettings.end_token

    df = pd.DataFrame({
        'text_left': table['QueryText'],
        'raw_text_left': table['QueryText'].copy(),

        'text_right': table['DocText'].copy(),  # used for testing
        'raw_text_right': table['DocText'].copy(),
        KeyWordSettings.TextRightInput: table['DocText'].copy().progress_apply(_prepare_decoder_input),  # for training
        KeyWordSettings.TextRightOutput: table['DocText'].copy().progress_apply(_prepare_decoder_target),  # for training

        'id_left': table['QueryID'],
        'id_right': table['DocID']
    })
    return pack.pack(df, selected_columns_left=['text_left', 'id_left', 'raw_text_left'],
                     selected_columns_right=['text_right', 'id_right', 'raw_text_right',
                                             KeyWordSettings.TextRightInput, KeyWordSettings.TextRightOutput])

