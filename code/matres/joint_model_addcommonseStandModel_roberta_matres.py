from pathlib import Path
import pickle
import sys
import argparse
from collections import defaultdict, Counter, OrderedDict
from itertools import combinations
from typing import Iterator, List, Mapping, Union, Optional, Set
import logging as log
import abc
import numpy as np
import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import Parameter
import math
import time
import copy
from torch.utils import data
from torch.nn.utils.rnn import pack_padded_sequence as pack, pad_packed_sequence as unpack
from functools import partial
from sklearn.model_selection import KFold, ParameterGrid, train_test_split
from utils import ClassificationReport
import logging
from FocalLoss import *
import os
torch.manual_seed(123)

tbd_label_map = OrderedDict([('VAGUE', 'VAGUE'),
                             ('BEFORE', 'BEFORE'),
                             ('AFTER', 'AFTER'),
                             ('SIMULTANEOUS', 'SIMULTANEOUS'),
                             ('INCLUDES', 'INCLUDES'),
                             ('IS_INCLUDED', 'IS_INCLUDED'),
                             ])

matres_label_map = OrderedDict([('VAGUE', 'VAGUE'),
                                ('BEFORE', 'BEFORE'),
                                ('AFTER', 'AFTER'),
                                ('SIMULTANEOUS', 'SIMULTANEOUS')
                                ])


class BertClassifier(nn.Module):
    """Neural Network Architecture"""

    def __init__(self, args):
        super(BertClassifier, self).__init__()
        self.hid_size = args.hid
        self.batch_size = args.batch
        self.num_layers = args.num_layers
        self.num_classes = len(args.label_to_id)
        self.num_ent_classes = 2

        self.dropout = nn.Dropout(p=args.dropout)
        # lstm is shared for both relation and entity
        #self.lstm = nn.LSTM(768, self.hid_size, self.num_layers, bias=False, bidirectional=True)
        self.lstm = nn.LSTM(1024, self.hid_size, self.num_layers, bias=False, bidirectional=True)

        # MLP classifier for relation
        self.linear1 = nn.Linear(self.hid_size * 4 + 2 * args.n_fts, self.hid_size)
        self.linear2 = nn.Linear(self.hid_size, self.num_classes)
        # self.linear2 = nn.Linear(self.hid_size + args.n_fts, self.num_classes)

        # MLP classifier for entity
        self.linear1_ent = nn.Linear(self.hid_size * 2, int(self.hid_size / 2))
        self.linear2_ent = nn.Linear(int(self.hid_size / 2), self.num_ent_classes)

        self.act = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.softmax_ent = nn.Softmax(dim=2)

        self.w_omega = nn.Parameter(torch.Tensor(1, args.hid * 2))
        self.u_omega = nn.Parameter(torch.Tensor(1, args.hid * 2))
        self.v_omega = nn.Parameter(torch.Tensor(args.hid * 2, 30))

        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        nn.init.uniform_(self.v_omega, -0.1, 0.1)

    # https://www.cnblogs.com/douzujun/p/13511237.html
    # def attention_net_tanh(x, query):
    # https://blog.csdn.net/qq_43613342/article/details/111343805
    def attn_tahn_network(self, Q, K):
        d_k = Q.size(0)
        u = torch.matmul(Q, self.w_omega)
        v = torch.matmul(K, self.u_omega)
        att = torch.add(u, v)
        att = torch.tanh(att) / math.sqrt(d_k)
        # att = torch.matmul(att, torch.t(att))
        att_score = F.softmax(att, dim=-1)
        # att_score = torch.matmul(att, torch.t(att_score))
        att_score = torch.matmul(att_score, torch.t(att))
        context = torch.matmul(att_score, Q)
        return context, att_score

    def forward(self, sents, lengths, fts=[], rel_idxs=[], lidx_start=[], lidx_end=[], ridx_start=[], ridx_end=[],
                pred_ind=True, flip=False, causal=False, token_type_ids=None, task='relation'):
        batch_size = sents.size(0)
        # dropout
        out = self.dropout(sents)
        # pack and lstm layer
        out, _ = self.lstm(pack(out, lengths, batch_first=True))
        # unpack
        out, _ = unpack(out, batch_first=True)

        # entity prediction
        if task == 'entity':
            out_ent = self.linear1_ent(self.dropout(out))
            out_ent = self.act(out_ent)
            out_ent = self.linear2_ent(out_ent)
            prob_ent = self.softmax_ent(out_ent)
            return out_ent, prob_ent
        # relation prediction-flatten hidden vars into a long vector
        if task == 'relation':
            ltar_f = torch.cat([out[b, lidx_start[b][r], :self.hid_size].unsqueeze(0) for b, r in rel_idxs], dim=0)
            ltar_b = torch.cat([out[b, lidx_end[b][r], self.hid_size:].unsqueeze(0) for b, r in rel_idxs], dim=0)

            rtar_f = torch.cat([out[b, ridx_start[b][r], :self.hid_size].unsqueeze(0) for b, r in rel_idxs], dim=0)
            rtar_b = torch.cat([out[b, ridx_end[b][r], self.hid_size:].unsqueeze(0) for b, r in rel_idxs], dim=0)

            out = self.dropout(torch.cat((ltar_f, ltar_b, rtar_f, rtar_b), dim=1))
            # out = torch.cat((out, fts), dim=1)

            disVec = fts[:, 0]
            disVec = disVec.unsqueeze(1)
            commonseVec1 = fts[:, 1]
            disVec = disVec.float()
            commonseVec1 = commonseVec1.tolist()

            # out = torch.cat((out, disVec), dim=1)

            commonseVec2 = fts[:, 2]
            commonseVec2 = commonseVec2.tolist()
            commonseVec = []

            for vec1, vec2 in zip(commonseVec1, commonseVec2):
                vec1 = round(vec1, 4)
                vec2 = round(vec2, 4)
                if vec1 > vec2:
                    commonseVec.append(1)
                elif vec1 < vec2:
                    commonseVec.append(2)
                elif vec1 == vec2 and vec1 != 0:
                    commonseVec.append(3)
                else:
                    commonseVec.append(0)
            commonseVec = torch.FloatTensor(commonseVec)
            commonseVec = commonseVec.unsqueeze(1)
            if args.cuda:
                commonseVec = commonseVec.cuda()
            # commonseVec = self.dropout(commonseVec)

            # addVec = commonseVec.mm(torch.t(disVec))
            # addVec = disVec.mm(torch.t(commonseVec))
            # prob_matrix
            # prob_matrix = F.softmax(addVec)
            # commonseVec = torch.matmul(prob_matrix, commonseVec)

            #commonseVec, _ = attention_net(disVec, commonseVec)
            commonseVec, _ = attention_net(commonseVec, disVec)
            #commonseVec, _ = self.attn_tahn_network(commonseVec, disVec)
            out, _ = attention_net(out, out)
            commonseVec = self.dropout(commonseVec)
            disVec = self.dropout(disVec)

            out = torch.cat((out, disVec, commonseVec), dim=1)
            # out = torch.cat((out, disVec), dim=1)
            # out = torch.cat((out, commonseVec), dim=1)

            # linear prediction
            out = self.linear1(out)
            out = self.act(out)
            out = self.dropout(out)

            # out = torch.cat((out, commonseVec), dim=1)

            # out = torch.cat((out, disVec), dim=1)
            out = self.linear2(out)
            prob = self.softmax(out)
            return out, prob
            print("##")


def attention_net(x, query):
    d_k = query.size(0)
    # scores = torch.matmul(query, x.transpose(1, 0)) / math.sqrt(d_k)
    scores = query.mm(torch.t(x)) / math.sqrt(d_k)
    p_attn = F.softmax(scores, dim=-1)
    context = torch.matmul(p_attn, x)
    return context, p_attn


class NNClassifier(nn.Module):
    def __init__(self):
        super(NNClassifier, self).__init__()

    def predict(self, model, data, args, test=False, gold=True, model_r=None):
        model.eval()
        criterion = nn.CrossEntropyLoss()
        count = 1
        labels, probs, losses_t, losses_e = [], [], [], []
        pred_ind, docs, pairs = [], [], []

        # storage non-predicted rels in list
        nopred_rels = []

        ent_pred_map, ent_label_map = {}, {}
        rd_pred_map, rd_label_map = {}, {}

        for doc_id, context_id, sents, ent_keys, ents, poss, rels, lengths in data:

            if args.cuda:
                sents = sents.cuda()
                ents = ents.cuda()

            # predict entity first
            out_e, prob_e = model(sents, lengths, task="entity")
            labels_r, fts, rel_idxs, doc, pair, lidx_start, lidx_end, ridx_start, ridx_end, nopred_rel = self.construct_relations(
                prob_e, lengths, rels, list(doc_id), poss, gold=gold)
            nopred_rels.extend(nopred_rel)

            ###predict relations
            if rel_idxs:
                docs.extend(doc)
                pairs.extend(pair)

                if args.cuda:
                    labels_r = labels_r.cuda()
                    fts = fts.cuda()

                if model_r:
                    model_r.eval()
                    out_r, prob_r = model_r(sents, lengths, fts=fts, rel_idxs=rel_idxs, lidx_start=lidx_start,
                                            lidx_end=lidx_end, ridx_start=ridx_start, ridx_end=ridx_end)
                else:
                    out_r, prob_r = model(sents, lengths, fts=fts, rel_idxs=rel_idxs, lidx_start=lidx_start,
                                          lidx_end=lidx_end, ridx_start=ridx_start, ridx_end=ridx_end)
                loss_r = criterion(out_r, labels_r)
                predicted = (prob_r.data.max(1)[1]).long().view(-1)

                if args.cuda:
                    loss_r = loss_r.cuda()
                    prob_r = prob_r.cuda()
                    labels_r = labels_r.cuda()
                    losses_t.append(loss_r.cuda().data.cpu().numpy())
                    probs.append(prob_r)
                    labels.append(labels_r)
                else:
                    losses_t.append(loss_r.data.numpy())
                    probs.append(prob_r)
                    labels.append(labels_r)
                """
                losses_t.append(loss_r.data.numpy())
                probs.append(prob_r)
                labels.append(labels_r)"""
            # retrieve and flatten entity prediction for loss calculation
            ent_pred, ent_label, ent_prob, ent_key, ent_pos = [], [], [], [], []
            for i, l in enumerate(lengths):
                # flatten prediction
                ent_pred.append(out_e[i, :l])
                # flatten entity prob
                ent_prob.append(prob_e[i, :l])
                # flatten entity label
                ent_label.append(ents[i, :l])
                # flatten entity key - a list of original (extend)
                assert len(ent_keys[i]) == l
                ent_key.extend(ent_keys[i])
                # flatten pos tags
                ent_pos.extend([p for p in poss[i]])

            ent_pred = torch.cat(ent_pred, 0)
            ent_label = torch.cat(ent_label, 0)
            ent_probs = torch.cat(ent_prob, 0)

            assert ent_pred.size(0) == ent_label.size(0)
            assert ent_pred.size(0) == len(ent_key)

            loss_e = criterion(ent_pred, ent_label)
            losses_e.append(loss_e.cpu().data.numpy())

            ent_label = ent_label.tolist()

            for i, v in enumerate(ent_key):
                label_e = ent_label[i]
                prob_e = ent_probs[i]

                # exclude sent_start and sent_sep
                if v in ["[SEP]", "[CLS]"]:
                    assert ent_pos[i] in ["[SEP]", "[CLS]"]

                if v not in ent_pred_map:
                    # only store the probability of being 1 (is an event)
                    ent_pred_map[v] = [prob_e.tolist()[1]]
                    ent_label_map[v] = (label_e, ent_pos[i])
                else:
                    # if key stored already,append another prediction
                    ent_pred_map[v].append(prob_e.tolist()[1])
                    # and ensure label is the same
                    assert ent_label_map[v][0] == label_e
                    assert ent_label_map[v][1] == ent_pos[i]

            count += 1
            if count % 10 == 0:
                logging.info("finished evaluating {} samples".format(count * args.batch))

                # print(doc_id)

        ## collect relation prediction results
        probs = torch.cat(probs, dim=0)
        labels = torch.cat(labels, dim=0)

        assert labels.size(0) == probs.size(0)
        # calculate entity F1 score here
        # update ent_pred map with [mean >0.5 -->1]
        ent_pred_map_agg = {k: 1 if np.mean(v) > 0.5 else 0 for k, v in ent_pred_map.items()}

        n_correct = 0
        n_pred = 0

        pos_keys = OrderedDict([(k, v) for k, v in ent_label_map.items() if v[0] == 1])
        n_true = len(pos_keys)

        for k, v in ent_label_map.items():
            if ent_pred_map_agg[k] == 1:
                n_pred += 1
            if ent_pred_map_agg[k] == 1 and ent_label_map[k][0] == 1:
                n_correct += 1
        print(n_pred, n_true, n_correct)

        def safe_division(numr, denr, on_err=0):
            return on_err if denr == 0.0 else float(numr) / float(denr)

        precision = safe_division(n_correct, n_pred)
        recall = safe_division(n_correct, n_true)
        f1_score = safe_division(2.0 * precision * recall, precision + recall)

        logging.info("Evaluation temporal relation loss:{}".format(np.mean(losses_t)))
        logging.info("Evaluation temporal entity loss:{},f1_score:{}".format(np.mean(losses_e), f1_score))

        if test:
            return probs.data, np.mean(losses_t), labels, docs, pairs, f1_score, nopred_rels
        else:
            return probs.data, np.mean(losses_t), labels, docs, pairs, n_pred, n_true, n_correct, nopred_rels

    def construct_relations(self, ent_probs, lengths, rels, doc, poss, gold=True, train=True):
        # many relation properties such rev and pred_int are not used for now
        nopred_rels = []
        ##Case 1:only use gold relation
        if gold:
            pred_rels = rels

        # relations are (flatten) lists of features
        # rel_idx indicates (batch_id,rel_id_batch_id)
        docs, pairs = [], []
        rel_idxs, lidx_start, lidx_end, ridx_start, ridx_end = [], [], [], [], []
        for i, rel in enumerate(pred_rels):
            rel_idxs.extend([(i, ii) for ii, _ in enumerate(rel)])
            lidx_start.append([x[5][0] for x in rel])
            lidx_end.append([x[5][1] for x in rel])
            ridx_start.append([x[5][2] for x in rel])
            ridx_end.append([x[5][3] for x in rel])
            pairs.extend([x[1] for x in rel])
            docs.extend([doc[i] for _ in rel])
        assert len(docs) == len(pairs)
        rels = [x for rel in pred_rels for x in rel]
        if rels == []:
            labels = torch.FloatTensor([])
            fts = torch.FloatTensor([])
        else:
            labels = torch.LongTensor([x[2] for x in rels])
            # fts = torch.cat([torch.FloatTensor(x[3]) for x in rels]).unsqueeze(1)
            add_rels = []
            for x in rels:
                common = x[3]
                if common.shape[1] != 3:
                    padcommon = torch.cat((common, torch.zeros(1, 3 - common.shape[1])), 1)
                    common = padcommon

                add_rels.append(common)
            fts = torch.cat(add_rels, dim=0)
        # print(rels)
        return labels, fts, rel_idxs, docs, pairs, lidx_start, lidx_end, ridx_start, ridx_end, nopred_rels

    def _train(self, train_data, eval_data, pos_emb, args):
        model = BertClassifier(args)

        if args.cuda:
            print("====================================using cuda device:%s====================================" % torch.cuda.current_device())
            print(
                "====================================using cuda device:%s====================================" % torch.cuda.current_device())
            print(
                "====================================using cuda device:%s====================================" % torch.cuda.current_device())
            assert torch.cuda.is_available()
            model.cuda()

        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        alpha1 = [0.1, 0.9]
        # alpha2 = [0.18, 2, 2, 7, 5, 5, 0.1]
        # alpha2 = [0.18, 1, 1, 6, 5, 5, 0.1]
        alpha2 = [0.1, 1, 1, 6, 0.1]

        criterion_e = FocalLoss(alpha=alpha1)
        # criterion_e = nn.CrossEntropyLoss()

        if args.data_type in ['tbd']:
            weights = torch.FloatTensor([1.0, 1.0, 1.0, args.uw, args.uw, args.uw, 1.0])
        elif args.data_type in ['comsense']:
            weights = torch.FloatTensor([1.0, 1.0, 1.0, args.uw, args.uw, args.uw, 1.0])
        else:
            weights = torch.FloatTensor([1.0, 1.0, 1.0, args.uw, 1.0])

        if args.cuda:
            weights = weights.cuda()

        # criterion_r = nn.CrossEntropyLoss(weight=weights)
        criterion_r = FocalLoss(alpha=alpha2)
        losses = []

        sents, poss, ftss, labels = [], [], [], []
        if args.load_model == True:
            checkpoint = torch.load(args.ilp_dir + args.entity_model_file)
            model.load_state_dict(checkpoint['state_dict'])
            epoch = checkpoint['epoch']
            best_eval_f1 = checkpoint['f1']
            logging.info("Local best eval f1 is:{%s}".format(best_eval_f1))
        best_eval_f1 = 0.0
        best_epoch = 0
        for epoch in range(args.epochs):
            logging.info("Training Epoch :{}".format(epoch))
            model.train()
            count = 1

            loss_hist_t, losses_hist_e = [], []
            start_time = time.time()

            gold = False if epoch > args.pipe_epoch else True
            for doc_id, context_id, sents, keys, ents, poss, rels, lengths in train_data:
                if args.cuda:
                    sents = sents.cuda()
                    ents = ents.cuda()

                model.zero_grad()

                # predict entity first
                out_e, prob_e = model(sents, lengths, task='entity')

                labels_r, fts, rel_idxs, doc, pair, lidx_start, lidx_end, ridx_start, ridx_end, nopred_rel = self.construct_relations(
                    prob_e, lengths, rels, list(doc_id), poss, gold=gold)
                if args.cuda:
                    labels_r = labels_r.cuda()
                    fts = fts.cuda()
                # retrieve and flatten entity prediction for loss calculation
                ent_pred, ent_label = [], []
                for i, l in enumerate(lengths):
                    # flatten prediction
                    ent_pred.append(out_e[i, :l])
                    ent_label.append(ents[i, :l])
                ent_pred = torch.cat(ent_pred, 0)
                ent_label = torch.cat(ent_label, 0)

                assert ent_pred.size(0) == ent_label.size(0)
                loss_e = criterion_e(ent_pred, ent_label)

                ##predict relations
                loss_r = 0
                if rel_idxs:
                    out_r, prob_r = model(sents, lengths, fts=fts, rel_idxs=rel_idxs, lidx_start=lidx_start,
                                          lidx_end=lidx_end, ridx_start=ridx_start, ridx_end=ridx_end)
                    loss_r = criterion_r(out_r, labels_r)
                total_loss = args.relation_weight * loss_r + args.entity_weight * loss_e
                total_loss.backward()
                optimizer.step()
                """
                if loss_r != 0:
                    loss_hist_t.append(loss_r.data.numpy())
                losses_hist_e.append(loss_e.data.numpy())
                """
                if args.cuda:
                    if loss_r != 0:
                        loss_hist_t.append(loss_r.cuda().data.cpu().numpy())
                    losses_hist_e.append(loss_e.cuda().data.cpu().numpy())
                else:
                    if loss_r != 0:
                        loss_hist_t.append(loss_r.data.numpy())
                    losses_hist_e.append(loss_e.data.numpy())

                if count % 100 == 0:
                    logging.info("trained :{} samples".format(count * args.batch))
                    logging.info("Temporal loss is {}".format(np.mean(loss_hist_t)))
                    logging.info("Entity loss is {}".format(np.mean(losses_hist_e)))
                    logging.info("{} seconds elapsed".format(time.time() - start_time))
                count += 1
            # Evaluate at the end of each epoch
            logging.info("*" * 50)
            if len(eval_data) > 0:
                # need to have a warm-start otherwise there could be no event_pred
                # may need to manually pick poch<#.but 0 generally works when ew is large
                eval_gold = gold
                eval_preds, eval_loss, eval_labels, _, _, ent_pred, ent_true, ent_corr, nopred_rels = self.predict(
                    model, eval_data, args, gold=eval_data)
                pred_labels = eval_preds.max(1)[1].long().view(-1)
                assert eval_labels.size() == pred_labels.size()

                eval_correct = (pred_labels == eval_labels).sum()
                eval_acc = float(eval_correct) / float(len(eval_labels))

                """
                pred_labels = list(pred_labels.numpy())
                eval_labels = list(eval_labels.numpy())"""

                if args.cuda:
                    pred_labels = list(pred_labels.cuda().data.cpu().numpy())
                    eval_labels = list(eval_labels.cuda().data.cpu().numpy())
                else:
                    pred_labels = list(pred_labels.numpy())
                    eval_labels = list(eval_labels.numpy())

                # Append non-predicted labels as label:Gold;Pred:None
                if not eval_gold:
                    print(len(nopred_rels))
                    pred_labels.extend([self._label_to_id['NONE'] for _ in nopred_rels])
                    eval_labels.extend(nopred_rels)

                if args.data_type in ['red', 'caters']:
                    pred_inds = []
                    pred_labels = [pred_labels[k] if v == 1 else self._label_to_id['NONE'] for k, v in
                                   enumerate(pred_inds)]
                # select model only based on entity +relation F1 score
                eval_f1 = self.weight_f1(pred_labels, eval_labels, ent_corr, ent_pred, ent_true,
                                         args.relation_weight, args.entity_weight)
                # args.pipe_epoch <=args.epochs if pipeline (joint) training is used
                if eval_f1 > best_eval_f1 and (epoch > args.pipe_epoch or args.pipe_epoch >= 1000):
                    best_eval_f1 = eval_f1
                    self.model = copy.deepcopy(model)
                    best_epoch = epoch
                logging.info("Evaluation loss:{};Evaluation F1:{}".format(eval_loss, eval_f1))
                logging.info("*" * 50)

                if len(eval_data) == 0 or args.load_model:
                    self.model = copy.deepcopy(model)
                    best_epoch = epoch

        return best_eval_f1, best_epoch

    def weight_f1(self, pred_labels, true_labels, ent_corr, ent_pred, ent_true, rw=0.0, ew=0.0):
        def safe_division(numr, denr, on_err=0.0):
            return on_err if denr == 0.0 else numr / denr

        assert len(pred_labels) == len(true_labels)

        weighted_f1_score = {}
        if "NONE" in self._label_to_id.keys():
            num_tests = len([x for x in true_labels if x != self._label_to_id['NONE']])
        else:
            num_tests = len([x for x in true_labels])

        logging.info("Total positive samples to eval:{}".format(num_tests))
        total_true = Counter(true_labels)
        total_pred = Counter(pred_labels)

        labels = list(self._id_to_label.keys())

        n_correct = 0
        n_true = 0
        n_pred = 0

        if rw > 0:
            # f1 score is used for tcr and matres and hence exclude vague
            exclude_labels = ['VAGUE', 'NONE'] if len(self._label_to_id) == 5 else ['NONE']

            for label in labels:
                if self._id_to_label[label] not in exclude_labels:
                    true_count = total_true.get(label, 0)
                    pred_count = total_pred.get(label, 0)

                    n_true += true_count
                    n_pred += pred_count

                    correct_count = len([l for l in range(len(pred_labels)) if
                                         pred_labels[l] == true_labels[l] and pred_labels[l] == label])
                    n_correct += correct_count

        if ew > 0:
            # add entity prediction results before calculating precision,recall and f1
            n_correct += ent_corr
            n_pred += ent_pred
            n_true += ent_true

        precision = safe_division(n_correct, n_pred)
        recall = safe_division(n_correct, n_true)
        f1_score = safe_division(2.0 * precision * recall, precision + recall)
        logging.info("Overall Precision:{},Recall:{},F1:{}".format(precision, recall, f1_score))
        return f1_score

    def train_epoch(self, train_data, dev_data, args, test_data=None):
        if args.data_type == "matres":
            label_map = matres_label_map
        if args.data_type == "tbd":
            label_map = tbd_label_map
        if args.data_type == "comsense":
            label_map = tbd_label_map
        assert len(label_map) > 0

        all_labels = list(OrderedDict.fromkeys(label_map.values()))
        ##append negative pair label
        all_labels.append('NONE')

        self._label_to_id = OrderedDict([(all_labels[l], l) for l in range(len(all_labels))])
        self._id_to_label = OrderedDict([(l, all_labels[l]) for l in range(len(all_labels))])

        print(self._label_to_id)
        print(self._id_to_label)

        args.label_to_id = self._label_to_id
        ###pos embedding is not used for now,but can be added later
        pos_emb = np.zeros((len(args.pos2idx) + 1, len(args.pos2idx) + 1))
        for i in range(pos_emb.shape[0]):
            pos_emb[i, i] = 1.0

        best_f1, best_epoch = self._train(train_data, dev_data, pos_emb, args)
        logging.info("Final Dev F1:{}".format(best_f1))
        # print("###", pos_emb)
        return best_f1, best_epoch


def pad_collate(batch):
    """
    Puts data,and lengths into a packed_padded_sequence the returns the packed_padded_sequence and the labels.set use_lengths to True
    to use this collate function
    Args:
        batch:(list of tuples)[(doc_id,sample_id,pair,label,sent,pos,fts,rev,lidx_start_s,lidx_end_s,ridx_start_s,ridx_end_s,pred_ind)].
    Output:
     pack_batch:(PackedSequence for sent and pos),see torch.nn.utils.rnn.pack_padded_sequence
     labels:(Tensor)
     other arguments remain the same.
    """
    if len(batch) >= 1:
        bs = list(zip(*[ex for ex in sorted(batch, key=lambda x: x[2].shape[0], reverse=True)]))
        max_len, n_fts = bs[2][0].shape
        lengths = [x.shape[0] for x in bs[2]]

        ###gather sents:idx =2 in batch_sorted
        sents = [torch.cat((torch.FloatTensor(s), torch.zeros(max_len - s.shape[0], n_fts)), 0) if s.shape[
                                                                                                       0] != max_len else torch.FloatTensor(
            s) for s in bs[2]]
        # print(sents)
        sents = torch.stack(sents, 0)

        # gather entity labels:idx = 3in batch_sorted
        # we need a unique doc_span key for aggregation later
        all_key_ent = [list(zip(*key_ent)) for key_ent in bs[3]]
        keys = [[(bs[0][i], k) for k in v[0]] for i, v in enumerate(all_key_ent)]
        ents = [v[1] for v in all_key_ent]
        ents = [torch.cat((torch.LongTensor(s).unsqueeze(1), torch.zeros(max_len - len(s), 1, dtype=torch.long)), 0)
                if len(s) != max_len else torch.LongTensor(s).unsqueeze(1) for s in ents]
        ents = torch.stack(ents, 0).squeeze(2)
    return bs[0], bs[1], sents, keys, ents, bs[4], bs[5], lengths


class EventDataSet(data.Dataset):
    """
    Characterizes a dataset for PyTorch
    """

    def __init__(self, data_dir, data_split):
        # load data
        with open(data_dir + data_split + '.pickle', 'rb') as handle:
            self.data = pickle.load(handle)
            self.data = list(self.data.values())
        handle.close()

    def __len__(self):
        """Denotes the total"""
        return len(self.data)

    def __getitem__(self, idx):
        """Generates one sample of data"""
        sample = self.data[idx]
        doc_id = sample['doc_id']
        context_id = sample['context_id']
        context = sample['context']
        rels = sample['rels']
        return doc_id, context_id, context[0], context[1], context[2], rels


def exec_func(args):
    data_dir = args.data_dir
    opt_args = {}
    #type_dir = "/all_context_comsense_agg/"
    type_dir = "/all_context_roberta_matres_agg/"
    # type_dir = "/all_context_comsense_agg_disaddcommonse/"
    train_data = EventDataSet(args.data_dir + type_dir, "train")
    params = {
        'batch_size': args.batch,
        'shuffle': False,
        'collate_fn': pad_collate,
    }
    add_data = EventDataSet(args.data_dir + type_dir, "dev")
    train_data = train_data + add_data
    train_generator = data.DataLoader(train_data, **params)

    #dev_data = EventDataSet(args.data_dir + type_dir, "dev")
    dev_data = EventDataSet(args.data_dir + type_dir, "test")
    dev_generator = data.DataLoader(dev_data, **params)

    test_data = EventDataSet(args.data_dir + type_dir, "test")
    test_generator = data.DataLoader(test_data, **params)

    model = NNClassifier()
    logging.info(f"======={args.model}=====\n")
    best_f1, best_epoch = model.train_epoch(train_generator, dev_generator, args)
    logging.info("Dev best_f1:{}, best_epoch:{}".format(best_f1, best_epoch))
    evaluator = EventEvaluator(model)
    rel_f1, ent_f1 = evaluator.evaluate(test_generator, args)
    logging.info("Test Set rel_f1:{}, ent_f1:{}".format(rel_f1, ent_f1))

    # print(train_generator)


class EventEvaluator:
    def __init__(self, model):
        self.model = model

    def evaluate(self, test_data, args):
        # load test data first since it needs to be excuted twice in this function
        logging.info("Start testing...")
        preds, loss, true_labels, docs, pairs, ent_f1, nopred_rels = self.model.predict(self.model.model, test_data,
                                                                                        args, test=True,
                                                                                        gold=args.eval_gold)
        preds = (preds.max(1)[1]).long().view(-1)

        if args.cuda:
            pred_labels = preds.cuda().data.cpu().numpy().tolist()
            true_labels = true_labels.cuda().data.cpu().numpy().tolist()
        else:
            pred_labels = preds.numpy().tolist()
            true_labels = true_labels.numpy().tolist()
        #pred_labels = preds.numpy().tolist()
        #true_labels = true_labels.numpy().tolist()
        if not args.eval_gold:
            print(len(nopred_rels))
            pred_labels.extend([self.model._label_to_id['NONE'] for _ in nopred_rels])
            true_labels.extend(nopred_rels)

        rel_f1 = self.model.weight_f1(pred_labels, true_labels, 0, 0, 0, rw=1)
        pred_labels = [self.model._id_to_label[x] for x in pred_labels]
        true_labels = [self.model._id_to_label[x] for x in true_labels]

        out = ClassificationReport(args.model, true_labels, pred_labels)
        print(out)
        print("F1 Excluding Vague: %.4f" % rel_f1)
        return rel_f1, ent_f1


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    p = argparse.ArgumentParser()
    # arguments for data processing
    p.add_argument('-data_dir', type=str, default="../data/")
    p.add_argument('-other_dir', type=str, default="../other/")
    # select model
    p.add_argument('-model', type=str, default='multitask/pipeline')
    # arguments for RNN model
    p.add_argument('-emb', type=int, default=100)
    p.add_argument('-hid', type=int, default=100)
    p.add_argument('-num_layers', type=int, default=1)
    p.add_argument('-batch', type=int, default=4)
    # p.add_argument('-data_type', type=str, default="comsense")
    p.add_argument('-data_type', type=str, default="matres")
    p.add_argument('-epochs', type=int, default=20)
    p.add_argument('-pipe_epoch', type=int, default=1000)
    p.add_argument('-seed', type=int, default=123)
    p.add_argument('-lr', type=float, default=1e-3)
    p.add_argument('-num_classes', type=int, default=2)
    p.add_argument('-dropout', type=float, default=0.3)
    #p.add_argument('-dropout', type=float, default=0.1)
    p.add_argument('-ngbr', type=int, default=15)
    p.add_argument('-pos2idx', type=dict, default={})
    p.add_argument('-w2i', type=OrderedDict)
    p.add_argument('-glove', type=OrderedDict)
    p.add_argument('-cuda',type=bool,default=True)
    p.add_argument('-refit_all', type=bool, default=False)
    p.add_argument('-uw', type=float, default=1.0)
    p.add_argument('-params', type=dict, default={})
    p.add_argument('-n_splits', type=int, default=5)
    p.add_argument('-pred_win', type=int, default=200)
    p.add_argument('-n_fts', type=int, default=1)
    p.add_argument('-relation_weight', type=float, default=1.0)
    p.add_argument('-entity_weight', type=float, default=3.0)
    #p.add_argument('-entity_weight', type=float, default=3.0)
    #p.add_argument('-entity_weight', type=float, default=10.0)
    p.add_argument('-save_model', type=bool, default=False)
    p.add_argument('-save_stamp', type=str, default="tbd_entity_best")
    p.add_argument('-entity_model_file', type=str, default="")
    p.add_argument('-relation_model_file', type=str, default="")
    p.add_argument('-load_model', type=bool, default=False)
    p.add_argument('-bert_config', type=dict, default={})
    p.add_argument('-fine_tune', type=bool, default=False)
    p.add_argument('-eval_gold', type=bool, default=True)

    args = p.parse_args()
    logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(levelname)s:%(message)s', level=logging.INFO,
                        datefmt='%I:%M:%S')
    args.save_stamp = "%s_hid%s_dropout%s_ew%s" % (args.save_stamp, args.hid, args.dropout, args.entity_weight)

    args.eval_list = []
    args.data_dir = args.data_dir + "/" + args.data_type
    # create pos_tag and vocabulary dictionary
    # make sure raw data files are stored in the same directory as train/dev/test data
    tags = open(args.other_dir + "/pos_tags.txt")
    pos2idx = {}
    # idx = 0
    for idx, tag in enumerate(tags):
        tag = tag.strip()
        pos2idx[tag] = idx
        # idx += 1
    args.pos2idx = pos2idx
    args.idx2pos = {v + 1: k for k, v in pos2idx.items()}
    print(args.hid, args.dropout, args.entity_weight, args.relation_weight)
    exec_func(args)
    print(args)
