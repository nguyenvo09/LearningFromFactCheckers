import torch
import torch.nn.functional as F
import torch.nn as nn
from Models.base_model import BaseModel
import torch_utils
from setting_keywords import KeyWordSettings
from Models.decoder import AttentiveDecoder
import time
import numpy as np
import torch.nn.utils.rnn as rnn_utils
from six.moves import cPickle
from torch.autograd import Variable
from handlers.output_handler import FileHandler


class FCRGModel(BaseModel):
    """ Fact-checking Response Generator (FCRG) """
    def __init__(self, params):
        super(BaseModel, self).__init__()
        self._params = params

        self.fixed_length_left = self._params["fixed_length_left"]
        self.fixed_length_right = self._params["fixed_length_right"]
        embedding_size = self._params['embedding_output_dim']
        embedding_dropout = self._params["embedding_dropout"]
        attention_type = self._params["attention_type"]

        hidden_size = self._params["hidden_size"]
        output_target_size = self._params["output_target_size"]
        bidirectional = self._params["bidirectional"]
        use_label = self._params["use_label"]
        use_input_feeding = self._params["use_input_feeding"]
        self.nlayers = self._params["nlayers"]
        # label_embedding_size = self._params["label_embedding_size"]  # True/False as label of classes

        self.embedding = self._make_default_embedding_layer(params)
        vocab_size = self.embedding.num_embeddings
        self.label_embedding = nn.Embedding(2, 128)

        self.embedding_dropout_layer = nn.Dropout(p=embedding_dropout)
        # this is for o-tweet
        self.encoder_gru = nn.GRU(embedding_size, hidden_size,
                               batch_first=True,
                               bidirectional=bidirectional,
                               num_layers=self.nlayers)

        self.decoder_att = AttentiveDecoder(attn_model=attention_type,
                                            hidden_size=hidden_size,
                                            output_size=output_target_size,
                                            embedding_layer=self.embedding,
                                            label_embedding_layer=self.label_embedding,
                                            bidirectional=bidirectional,
                                            dropout=embedding_dropout,
                                            n_layers=self.nlayers,
                                            use_label=use_label,
                                            use_input_feeding=use_input_feeding)
        num_directions = self.nlayers
        # if bidirectional == True: num_directions = 2 * num_directions
        # for top of decoder
        self.outputs2vocab = nn.Linear(output_target_size, vocab_size)

    def encoder(self, encoder_input: torch.Tensor, encoder_input_length: torch.Tensor, encoder_decoder_veracities = None,
                message = None, **kargs):
        """
        Encoder part, since we have varied lengths of sequences and we need to get outputs of every time step,
        we need pad_packed_sequence in PyTorch.
        :param encoder_input:
        :param encoder_input_length:
        :param encoder_decoder_veracities:
        :param message:
        :param kargs:
        :return:
        """
        assert KeyWordSettings.Query_lens in kargs and KeyWordSettings.Doc_lens in kargs
        assert encoder_input.size(0) == encoder_input_length.size(0)
        q_new_indices, q_restoring_indices, q_lens = kargs[KeyWordSettings.QueryLensIndices]
        # batch_size = encoder_input.size(0)
        encoder_otweet_embedding = self.embedding(encoder_input)
        encoder_otweet_embedding = self.embedding_dropout_layer(encoder_otweet_embedding)
        outputs, last_hidden_state = torch_utils.rnn_forward(self.encoder_gru,
                                                             (encoder_otweet_embedding, q_lens, q_new_indices, q_restoring_indices),
                                                             return_h=True, max_len=self.fixed_length_left)
        assert outputs.size(0) == last_hidden_state.size(0)  # (B, L, H) vs (B, H)
        last_hidden_state = last_hidden_state.unsqueeze(0)  # (1, B, H)
        assert len(last_hidden_state.size()) == 3
        return outputs, last_hidden_state

    def write_pickle_file(self, filename, obj):
        with open(filename, "wb") as fout:
            cPickle.dump(obj, fout)

    def decoder(self, decoder_inputs: torch.Tensor, init_states: torch.Tensor,
                encoder_outputs: torch.Tensor, encoder_decoder_veracities, **kargs):
        """
        Decoder with attention over pad_packed_sequence outputted from encoder. We need to loop step by step
        in the decoder. Usually, we can input the whole sequence into the decoder and let it run quickly.
        However, we are now using attention mechanism to derive context of each time-step in decoder.
        Therefore, we need to loop step-by-step based on input sequence of decoder.

        :param decoder_inputs: shape (batch_size, seq_length)
        :param init_states: This is the last hidden states outputted from Encoder
        :param encoder_outputs: This is pad_packed_sequence tensor with size (batch_size, seq_len, hidden_size)
        :param decoder_input_lengths: shape (batch_size, 1)
        :return:
        """
        batch_size = decoder_inputs.size(0)
        max_target_length = decoder_inputs.size(1) # seq_length of d-tweets

        all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, self.decoder_att.output_size))
        use_cuda = kargs[KeyWordSettings.UseCuda]
        all_decoder_outputs = torch_utils.gpu(all_decoder_outputs, use_cuda)

        decoder_hidden = init_states
        input_feed_init = torch.zeros(batch_size, self.decoder_att.hidden_size)
        input_feed_init = torch_utils.gpu(input_feed_init, use_cuda)

        # we start from 0
        for t in range(max_target_length):
            tic = time.time()
            decoder_input = decoder_inputs[:, t]  # Next input is current target
            # we need to change hidden_state after each step.
            decoder_output, decoder_hidden, decoder_attn, input_feed_init = self.decoder_att(
                decoder_input, decoder_hidden, encoder_outputs, encoder_decoder_veracities, input_feed_init)

            toc = time.time()
            all_decoder_outputs[t] = decoder_output # shape (batch_size, self.decoder_att.output_size)

        all_decoder_outputs = all_decoder_outputs.permute(1, 0, 2) # (B, max_target_length, decoder_output_size)
        logits = self.outputs2vocab(all_decoder_outputs)
        return logits, decoder_hidden

    def forward(self, encoder_input: torch.Tensor, decoder_input: torch.Tensor,
                encoder_input_length: torch.Tensor, encoder_decoder_veracities, **kargs):
        """
        Using pack-padded-sequence.

        :param encoder_input: PyTorch Tensor
        :param decoder_input: pytorch_tensor (batch_size, max_len_dweets) PyTorch Tensor
        :param encoder_input_length: shape = (batch_size, max_len_oritweets) PyTorch Tensor
        :param encoder_decoder_veracities: (B, )
        :return: logits
        """

        batch_size, max_seq_len_d_tweets = decoder_input.size()
        assert encoder_input.size(0) == batch_size
        encoder_outputs, last_hidden_state = self.encoder(encoder_input, encoder_input_length, encoder_decoder_veracities, **kargs)
        logits, next_states = self.decoder(decoder_input, last_hidden_state, encoder_outputs, encoder_decoder_veracities, **kargs)
        return logits

    def learnable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad == True]
