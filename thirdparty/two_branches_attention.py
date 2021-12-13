from torch import nn
import torch
import torch.nn.functional as F


class DotAttention(nn.Module):
    def __init__(self, method, hidden_size, attention_size):
        super(DotAttention, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attention_size = attention_size

    def forward(self, hidden, encoder_outputs):
        """
        :param hidden: shape = (B, seq_len = 1, H). This is the hidden_state output from a time-step in decoder
        :param encoder_outputs: shape = (B, L, H) outputs from every time steps in encoder
        :return:
        """
        batch_size, encoder_sequence_length, hidden_size = encoder_outputs.size()
        assert hidden.size(0) == encoder_outputs.size(0), "Hidden size: %s vs. encoder_outputs Size: %s" % \
                                                          (str(hidden.size()), str(encoder_outputs.size()))  # (B, )
        s = self.score(hidden, encoder_outputs)  # (B, 1, L)
        assert s.size() == (batch_size, 1, encoder_sequence_length), "Here: %s" % str(s.size())
        return F.softmax(s, dim = -1)  # (B, 1, L)

    def score(self, hidden: torch.Tensor, encoder_outputs: torch.Tensor):
        """
        Compute attention weight of a given hidden_state at a time-step of D-tweet
         with respect to an encoder_output_hidden_state in query
        :param hidden: shape (B, 1, H) (batch_size, seq_len = 1, hidden_size) of only one sequence. 1D tensor
        :param encoder_output: (B, L, H)
        :return:
        """
        assert self.method == "dot"
        energy = torch.bmm(hidden, encoder_outputs.permute(0, 2, 1))  # (B, 1, L)
        return energy
