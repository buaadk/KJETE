import torch
import torch.nn as nn
import torch.nn.functional as F


class lstm_siam(nn.Module):  # add categorical emb to two places instead of one
    def __init__(self, params, bigramGetter, granularity=0.05, common_sense_emb_dim=64, bidirectional=False,
                 lowerCase=False, verb_i_map=None):
        super(lstm_siam, self).__init__()
        self.params = params
        # self.emb_cache = emb_cache
        self.bigramGetter = bigramGetter

    def forward(self, word1, word2):
        # common sense embeddings
        bigramstats = self.bigramGetter.getBigramStatsFromTemprob(word1, word2)
        return bigramstats
