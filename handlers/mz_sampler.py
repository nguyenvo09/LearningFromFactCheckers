"""
Module containing functions for negative item sampling.
"""

import numpy as np
from scipy.sparse import csr_matrix
import torch_utils
import time
import interactions


class Sampler(object):
    def __init__(self):
        super(Sampler, self).__init__()
        self._candidate = dict()  # negative candidates

    def get_instances(self, interactions: interactions.MatchInteraction):
        """
        Sample negative from a candidate set of each user. The
        candidate set of each user is defined by:
        {All Items} \ {Items Rated by User}
        Parameters
        ----------
        interactions: :class:`matchzoo.DataPack`
            training instances, used for generate candidates. Note that
            since I am using MatchZoo datapack, there are negative cases in left-right relation ship as
            well.
        num_negatives: int
            total number of negatives to sample for each sequence
        """

        query_ids = interactions.pos_queries.astype(np.int64)  # may not be unique
        query_contents = interactions.np_query_contents.astype(np.int64)
        query_lengths = interactions.np_query_lengths.astype(np.int64)

        doc_ids = interactions.pos_docs.astype(np.int64)
        doc_input_contents = interactions.np_doc_decoder_input_contents.astype(np.int64)
        doc_output_contents = interactions.np_doc_decoder_output_contents.astype(np.int64)
        doc_lengths = interactions.np_doc_lengths.astype(np.int64) + 1  # due to <START> and <EOS> token

        target_contents = doc_output_contents
        return query_ids, query_contents, query_lengths, \
               doc_ids, doc_input_contents, target_contents, doc_lengths

