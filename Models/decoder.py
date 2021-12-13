import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import numpy as np
from thirdparty.two_branches_attention import DotAttention
from handlers.output_handler import FileHandler


class AttentiveDecoder(nn.Module):
    """ Attentive decoder """
    def __init__(self, attn_model, hidden_size: int, output_size: int, embedding_layer,
                 label_embedding_layer,
                 bidirectional: bool = False,
                 n_layers: int = 1, dropout: float = 0.1,
                 use_self_att = False, use_label: bool = False, use_input_feeding: bool = False):
        super(AttentiveDecoder, self).__init__()
        FileHandler.myprint("Using Attention type : %s" % attn_model)
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.embedding = embedding_layer
        embedding_size = embedding_layer.embedding_dim

        self.label_embedding = label_embedding_layer
        self.embedding_size = embedding_size

        self.use_label = use_label  # for label embedding
        self.label_embedding_size = 0
        self.embedding_dropout = nn.Dropout(dropout)
        self.input_feeding_size = 0
        self.use_input_feeding = use_input_feeding
        if self.use_input_feeding:
            self.input_feeding_size = self.hidden_size # we want it to be same as hidden_size

        self.gru = nn.GRU(self.embedding_size + self.input_feeding_size, self.hidden_size,
                          batch_first = True, bidirectional = self.bidirectional, num_layers = self.n_layers)

        num_direct = 1
        self.concat = nn.Linear(hidden_size * 2 * num_direct, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.atten_faster = DotAttention(attn_model, num_direct * hidden_size, hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, encoder_labels, input_feeding_prev_time_step):
        """
        Running decoder step by step
        :param input_seq: shape (B, )
        :param last_hidden: including (hidden_state, cell_state)
        :param encoder_outputs: (B, L, H)
        :param input_feeding_prev_time_step: shape (B, self.input_feeding_size)
        :return: output shape (B, Output_size),
        (last_hidden_state, last_cell_state)
        attn_weights with shape (B, S=1, time_steps_in_encoder)
        """
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded) # (batch_size, embedding_size)

        if self.use_input_feeding:
            embedded = torch.cat([embedded, input_feeding_prev_time_step], dim = -1)

        # embedded = embedded.view(1, batch_size, self.embedding_size)  # S=1 x B x N
        embedded = embedded.unsqueeze(1)  # Shape = B x 1 x embedding_size, indicating seq_len = 1.

        # Get current hidden state from input word and last hidden state
        # rnn_output, hidden = self.gru(embedded, last_hidden)
        # We have rnn_output.shape = (batch_size, seq_len = 1, num_direct * hidden_size = 1 * hidden_size)
        last_hidden = last_hidden.contiguous()
        rnn_output, last_hidden_state = self.gru(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.atten_faster(rnn_output, encoder_outputs)  # (B, seq_len = 1, time_steps_in_encoder)
        context = attn_weights.bmm(encoder_outputs)  # shape_of_context = (B, decoder_seq_len = 1, hidden_size)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(1)  # B x S=1 x N -> B x N
        context = context.squeeze(1)  # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), dim = -1)
        concat_output = F.tanh(self.concat(concat_input))

        if self.use_input_feeding: input_feeding_prev_time_step = concat_output
        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)

        # Return final output, hidden state, and attention weights (for visualization)
        return output, last_hidden_state, attn_weights, input_feeding_prev_time_step
