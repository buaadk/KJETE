import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


class lstm_siam(nn.Module):
    def __init__(self, params, bigramGetter, granularity=0.05, common_sense_emb_dim=64, bidirectional=False,
                 lowerCase=False, verb_i_map=None):
        super(lstm_siam, self).__init__()
        self.params = params
        self.embedding_dim = params.get('embedding_dim')
        self.lstm_hidden_dim = params.get('lstm_hidden_dim', 64)
        self.nn_hidden_dim = params.get('nn_hidden_dim', 32)
        self.bigramStats_dim = params.get('bigramStats_dim')
        self.bigramGetter = bigramGetter
        self.output_dim = params.get('output_dim', 4)
        self.batch_size = params.get('batch_size', 1)
        self.granularity = granularity
        self.common_sense_emb_dim = common_sense_emb_dim
        #self.common_sense_emb = nn.Embedding(487, common_sense_emb_dim)
        self.all_verbs_len = params.get('all_verbs_len', 487)
        self.common_sense_emb = nn.Embedding(self.all_verbs_len, common_sense_emb_dim)
        self.verb_i_map = verb_i_map


        self.sent_emb = nn.Embedding(self.all_verbs_len, self.embedding_dim)
        self.bidirectional = bidirectional
        self.lowerCase = lowerCase
        if self.bidirectional:
            self.lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden_dim // 2, num_layers=1, bidirectional=True)
        else:
            self.lstm = nn.LSTM(self.embedding_dim, self.lstm_hidden_dim, num_layers=1, bidirectional=False)
        self.h_lstm2h_nn = nn.Linear(384, self.output_dim)
        self.h_nn2o = nn.Linear(132, self.output_dim)

    def init_hidden(self):
        if self.bidirectional:
            self.hidden = (torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2), \
                           torch.randn(2 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim // 2))
        else:
            self.hidden = (torch.randn(1 * self.lstm.num_layers, self.batch_size, self.lstm_hidden_dim),
                           torch.randn(1 * self.lstm_hidden_dim, self.batch_size, self.lstm_hidden_dim))

    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.h_lstm2h_nn.reset_parameters()
        self.h_nn2o.reset_parameters()

    def forward(self, word1,word2):
        self.init_hidden()
        # common sense embeddings
        bigramstats = self.bigramGetter.getBigramStatsFromTemprelKnowledge(word1,word2)

        common_sense_emb = torch.FloatTensor(bigramstats.detach().numpy())

        return common_sense_emb

    def forwardcommon_sense(self, word1,word2):
        self.init_hidden()
        # common sense embeddings
        bigramstats = self.bigramGetter.getBigramStatsFromTemprelKnowledge(word1,word2)

        commonTensor = torch.LongTensor(bigramstats.detach().numpy())
        common_sense_emb = self.common_sense_emb(commonTensor)
        common_sense_emb = common_sense_emb.view(1, -1)

        #print("##lstm_siam##", common_sense_emb)
        return common_sense_emb

    def forwardSiam(self, temprel):
        self.init_hidden()
        # common sense embeddings
        bigramstats = self.bigramGetter.getBigramStatsFromTemprelKnowledge(temprel)
        # bigramTensor = bigramstats[0]
        # bigramTensor0 = bigramTensor[0]
        # bigramTensor1 = bigramTensor[1]
        commonTensor = torch.LongTensor(bigramstats.detach().numpy())
        common_sense_emb = self.common_sense_emb(commonTensor)
        common_sense_emb = common_sense_emb.view(1, -1)

        tokenList = temprel.token
        vecList = []
        for word in tokenList:
            if word not in self.verb_i_map.keys():
                v2vec = 0
            else:
                v2vec = self.verb_i_map[word]
                v2vec = v2vec % 487
            vecList.append(v2vec)
        sentList = torch.from_numpy(np.array(vecList))
        embed = self.sent_emb(sentList)
        embeds = embed.view(temprel.length, self.batch_size, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)

        firstDim = embeds.size()
        firstDim = firstDim [0]
        lstm_out = lstm_out.view(firstDim, self.batch_size, self.lstm_hidden_dim)
        event_id = temprel.event_ix
        lstm_out = lstm_out[event_id][:][:]
        lstm_out = lstm_out.view(1,-1)
        lstmcat = torch.cat((lstm_out,common_sense_emb),1)
        h_lstm2h = self.h_lstm2h_nn(lstmcat)
        h_nn = F.relu(h_lstm2h)

        #print("##lstm_siam##", h_nn.shape)
        return h_nn
