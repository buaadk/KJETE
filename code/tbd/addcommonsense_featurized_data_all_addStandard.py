import pickle
from typing import Tuple
import argparse
from collections import defaultdict, Counter, OrderedDict
import random
import logging as log
from pytorch_pretrained_bert.modeling import BertModel, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
import os
import torch
from torch.utils import data
import time
# from featureFuncs import *
from featureFuncsExec import *
from sklearn.model_selection import train_test_split
from myCommonseSelfLSTM import lstm_siam
from pairwise_ffnetwork_pytorch import VerbNet

tbd_label_map = OrderedDict([
    ('VAGUE', 'VAGUE'),
    ('BEFORE', 'BEFORE'),
    ('AFTER', 'AFTER'),
    ('SIMULTANEOUS', 'SIMULTANEOUS'),
    ('INCLUDES', 'INCLUDES'),
    ('IS_INCLUDED', 'IS_INCLUDED')
])

matres_label_map = OrderedDict([
    ('VAGUE', 'VAGUE'),
    ('BEFORE', 'BEFORE'),
    ('AFTER', 'AFTER'),
    ('SIMULTANEOUS', 'SIMULTANEOUS')
])


class Event():
    id: str
    type: str
    text: str
    tense: str
    polarity: str
    span: (int, int)


def data_split(train_docs, eval_docs, data, neg_r=0.0, seed=7):
    train_set = []
    eval_set = []
    train_set_neg = []

    for s in data:
        # dev-set doesn't require unlabled data
        if s[0] in eval_docs:
            # 0:doc_id, 1:ex.id, 2:(ex.left.id, ex.right.id), 3:label_id, 4:features
            eval_set.append(s)
        elif s[1][0] in ['L', 'C']:
            train_set.append(s)
        elif s[1][0] in ['N']:
            train_set_neg.append(s)

    random.Random(seed).shuffle(train_set_neg)
    n_neg = int(neg_r * len(train_set))
    if n_neg > 0:
        train_set.extend(train_set_neg[:n_neg])
        random.Random(seed).shuffle(train_set)

    return train_set, eval_set


def split_and_save(train_docs, dev_docs, data, seed, save_dir, nr=0.0):
    # first split labeled into train and dev
    train_data, dev_data = data_split(train_docs, dev_docs, data, neg_r=nr)
    print(len(train_data), len(dev_data))

    # shuffle
    # random.Random(seed).shuffle(train_data)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    with open(save_dir + '/train.pickle', 'wb') as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

    with open(save_dir + '/dev.pickle', 'wb') as handle:
        pickle.dump(dev_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

    return


def create_feature(ex, posidx, w2i, tokenizer, bert_model, lstmsiammodel):
    # bert_model.eval()
    pos_dict = ex['doc_dictionary']
    ent_labels = ex['event_labels']
    """定位位置"""
    all_keys, lidx_start, lidx_end, ridx_start, ridx_end = \
        token_idx(ex['left_event'].span, ex['right_event'].span, pos_dict)

    # truncate dictionary into three pieces
    left_seq = [pos_dict[x][0] for x in all_keys[:lidx_start]]
    right_seq = [pos_dict[x][0] for x in all_keys[ridx_end + 1:]]
    in_seq = [pos_dict[x][0] for x in all_keys[lidx_start:ridx_end + 1]]

    # find context sentence(s) start and end indices 找到句子的开始和结束
    try:
        sent_start = max(loc for loc, val in enumerate(left_seq) if val == ".") + 1
    except:
        sent_start = 0

    try:
        sent_end = ridx_end + 1 + min(loc for loc, val in enumerate(right_seq) if val == ".")
    except:
        sent_end = len(pos_dict)

    assert sent_start < sent_end
    assert sent_start <= lidx_start
    assert ridx_end <= sent_end

    # if >2 sentences,not predicting
    pred_ind = True
    if len([x for x in in_seq if x == "."]) > 1:
        pred_ind = False

    sent_key = all_keys[sent_start:sent_end]

    orig_sent = [pos_dict[x][0].lower() for x in sent_key]

    sent = [args.w2i[t] if t in args.w2i.keys() else 1 for t in orig_sent]

    pos = [pos2idx[k] if k in pos2idx.keys() else len(pos2idx) for k in [pos_dict[x][1] for x in sent_key]]

    ent = [(x, ent_labels[x]) for x in sent_key]

    # calculate events's index in context sentences
    lidx_start_s = lidx_start - sent_start
    lidx_end_s = lidx_end - sent_start
    ridx_start_s = ridx_start - sent_start
    ridx_end_s = ridx_end - sent_start

    # bert sentence segment ids
    segments_ids = []  # [0,.......0,1,.......1]
    seg = 0
    bert_pos = []
    bert_ent = []

    # append sentence start
    bert_tokens = ["[CLS]"]
    # original token to bert word-piece token mapping
    orig_to_tok_map = []

    segments_ids.append(seg)
    bert_pos.append("[CLS]")

    # sent start is non-event by default
    bert_ent.append(("[CLS]", 0))

    for i, token in enumerate(orig_sent):
        orig_to_tok_map.append(len(bert_tokens))
        if token == ".":
            segments_ids.append(seg)
            bert_pos.append("[SEP]")
            if seg == 0:
                seg = 1
                bert_tokens.append("[SEP]")
            else:
                bert_tokens.append(".")
            # sentence sep is no-event by default
            bert_ent.append(("[SEP]", 0))
        else:
            temp_tokens = tokenizer.tokenize(token)
            bert_tokens.extend(temp_tokens)
            for t in temp_tokens:
                segments_ids.append(seg)
                bert_pos.append(pos[i])
                bert_ent.append(ent[i])
    orig_to_tok_map.append(len(bert_tokens))

    bert_tokens.append("[SEP]")
    bert_pos.append("[SEP]")
    bert_ent.append(("[SEP]", 0))

    segments_ids.append(seg)
    assert len(segments_ids) == len(bert_tokens)
    assert len(bert_pos) == len(bert_tokens)

    # map original token index into bert (word_piece) index
    lidx_start_s = orig_to_tok_map[lidx_start_s]
    lidx_end_s = orig_to_tok_map[lidx_end_s + 1] - 1

    ridx_start_s = orig_to_tok_map[ridx_start_s]
    ridx_end_s = orig_to_tok_map[ridx_end_s + 1] - 1

    bert_sent = tokenizer.convert_tokens_to_ids(bert_tokens)

    bert_sent = torch.tensor([bert_sent])
    segs_sent = torch.tensor([segments_ids])

    # use the last layer computed by BERT as token vectors
    try:
        out, _ = bert_model(bert_sent, segs_sent)
        sent = out[-1].squeeze(0).data.numpy()
    except:
        sent_len = len(bert_tokens)
        sent = []
        bert_pos = []
    # create lexical features for the model
    #news_fts = []
    #news_fts.append(-distance_features(lidx_start, lidx_end, ridx_start, ridx_end))
    dismsg = -distance_features(lidx_start, lidx_end, ridx_start, ridx_end)
    fts = []
    fts.append(dismsg)
    fts = np.array(fts)
    dismsg = torch.from_numpy(fts).unsqueeze(1)
    # fts = torch.cat([torch.FloatTensor(x[3]) for x in rels]).unsqueeze(1)

    word1 = ex['left_event'].text
    word2 = ex['right_event'].text
    commonSense = lstmsiammodel(word1, word2)

    ftsMsg = torch.cat([dismsg, commonSense], dim=1)

    #return (sent, bert_ent, bert_pos, news_fts, ex['rev'], lidx_start_s, lidx_end_s, ridx_start_s, ridx_end_s, pred_ind)
    return (sent, bert_ent, bert_pos, ftsMsg, ex['rev'], lidx_start_s, lidx_end_s, ridx_start_s, ridx_end_s, pred_ind)


def parallel(ex, ex_id, args, tokenizer, bert_model):
    label_id = args._label_to_id[ex['rel_type']]
    bigramGetter = args.bigramGetter
    all_verbs_len = args.all_verbs_len
    params = {
        'embedding_dim': 1024,
        'lstm_hidden_dim': 128,
        'nn_hidden_dim': 64,
        'position_emb_dim': 32,
        'bigramStats_dim': 1,
        'lemma_emb_dim': 200,
        'dropout': False,
        'batch_size': 1,
        'all_verbs_len': all_verbs_len
    }
    lstmsiammodel = lstm_siam(params, bigramGetter=bigramGetter, granularity=0.1,
                              common_sense_emb_dim=64,
                              bidirectional=True, lowerCase=False, verb_i_map=bigramGetter.verb_i_map)

    return ex['doc_id'], ex_id, (ex['left_event'].id, ex['right_event'].id), label_id, \
           create_feature(ex, args.pos2idx, args.w2i, tokenizer, bert_model, lstmsiammodel)


def main(args):
    # to pick up here
    if args.data_type == "matres":
        label_map = matres_label_map
    elif args.data_type == "tbd":
        label_map = tbd_label_map
    elif args.data_type == "comsense":
        label_map = tbd_label_map

    all_labels = list(OrderedDict.fromkeys(label_map.values()))

    args._label_to_id = OrderedDict([(all_labels[l], l) for l in range(len(all_labels))])
    args._id_to_label = OrderedDict([(l, all_labels[l]) for l in range(len(all_labels))])

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    if args.load_model_dir:
        output_model_file = os.path.join(args.load_model_dir, "pytorch_model.bin")
        model_state_dict = torch.load(output_model_file)
        bert_model = BertModel.from_pretrained("../bert-base-uncased/", state_dict=model_state_dict)
    else:
        bert_model = BertModel.from_pretrained("../bert-base-uncased/")

    train_data = pickle.load(open(args.data_dir + "/train.pickle", "rb"))
    print("process train ...")
    data = [parallel(v, k, args, tokenizer, bert_model) for k, v in train_data.items()]

    if args.data_type in ['tbd', 'comsense']:
        print("process dev ...")
        dev_data = pickle.load(open(args.data_dir + "/dev.pickle", "rb"))
        dev_data = [parallel(v, k, args, tokenizer, bert_model) for k, v in dev_data.items()]
        data += dev_data

    # doc splits
    if args.data_type in ['matres']:
        train_docs, dev_docs = train_test_split(args.train_docs, test_size=0.2, random_state=args.seed)
    # TBDense data has given splits on train/dev/test
    else:
        train_docs = args.train_docs
        dev_docs = args.dev_docs

    if not os.path.isdir(args.save_data_dir):
        os.mkdir(args.save_data_dir)

    if "all" in args.split:
        print("process test...")
        # 读取数据集
        test_data = pickle.load(open(args.data_dir + "/test.pickle", "rb"))
        test_data = [parallel(v, k, args, tokenizer, bert_model) for k, v in test_data.items()]

        # 保存数据集
        with open(args.save_data_dir + "/test.pickle", "wb") as handle:
            pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

        print("###split_and_save###")

        split_and_save(train_docs, dev_docs, data, args.seed, args.save_data_dir)

        # quick trick to reduce number of tokens in GloVe

    return


class bigramGetter_fromNN3:
    def __init__(self, emb_path, mdl_path, ratio=0.3, layer=1, emb_size=200, splitter=','):
        self.verb_i_map = {}
        f = open(emb_path)
        lines = f.readlines()
        for i, line in enumerate(lines):
            msg = line.split(splitter)
            msg = msg[0]
            self.verb_i_map[msg] = i
        f.close()

        hidden_ratio = 128
        # emb_size = 128
        # all_verbs_len = 487
        self.all_verbs_len = len(self.verb_i_map)
        #self.all_verbs_len = 487
        # emb_size =128
        # self.model = VerbNet(all_verbs_len, hidden_ration=hidden_ratio, emb_size=emb_size, num_layers=layer)
        self.model = VerbNet(self.all_verbs_len, hidden_ration=ratio, emb_size=emb_size, num_layers=layer)
        checkpoint = torch.load(mdl_path, map_location='cpu')
        # self.model.load_state_dict(checkpoint)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        print("end")

    def eval(self, v1, v2):
        v1vec = self.verb_i_map[v1]
        v2vec = self.verb_i_map[v2]
        # v1vec = v1vec % 479
        # v2vec = v2vec % 479
        varray = [v1vec, v2vec]
        queryArr = []
        queryArr.append(varray)
        return self.model(torch.from_numpy(np.array(queryArr)))

    def getBigramStatsFromTemprelKnowledge(self, word1, word2):
        v1, v2 = word1, word2

        if v1 not in self.verb_i_map or v2 not in self.verb_i_map:
            return torch.FloatTensor([0.0]).view(1, -1)
        ent1vec = self.eval(v1, v2)
        ent2vec = self.eval(v2, v1)
        catVec = torch.cat((ent1vec, ent2vec), 1)
        catVecRen = catVec.view(1, -1)
        return catVecRen


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-data_dir', type=str, default='../data/')
    p.add_argument('-other_dir', type=str, default='../other/')
    p.add_argument('-load_model_dir', type=str, default='')
    p.add_argument('-train_docs', type=list, default=[])
    p.add_argument('-dev_docs', type=list, default=[])
    p.add_argument('-split', type=str, default='bert_all_joint_commonse')
    # p.add_argument('-data_type', type=str, default='matres')
    p.add_argument('-data_type', type=str, default='comsense')
    p.add_argument('-seed', type=int, default=7)
    args = p.parse_args()

    args.data_dir = args.data_dir + "/" + args.data_type
    if args.data_type == "tbd":
        args.train_docs = [x.strip() for x in open("%s/train_docs.txt" % args.data_dir, 'r')]
        args.dev_docs = [x.strip() for x in open("%s/dev_docs.txt" % args.data_dir, 'r')]
    elif args.data_type == "comsense":
        args.train_docs = [x.strip() for x in open("%s/train_docs.txt" % args.data_dir, 'r')]
        args.dev_docs = [x.strip() for x in open("%s/dev_docs.txt" % args.data_dir, 'r')]
    elif args.data_type == "matres":
        args.train_docs = [x.strip() for x in open("%s/train_docs.txt" % args.data_dir, 'r')]
    args.save_data_dir = args.data_dir + '/' + args.split

    glove = read_glove(args.other_dir + "/glove.6B.50d.txt")
    # glove = read_glove(args.other_dir + "/glove.6B.50d2.txt")
    vocab = np.array(['<pad>', '<unk>'] + list(glove.keys()))
    args.w2i = OrderedDict((vocab[i], i) for i in range(len(vocab)))

    tags = open(args.other_dir + "/pos_tags.txt")
    pos2idx = {}
    idx = 0
    for tag in tags:
        tag = tag.strip()
        pos2idx[tag] = idx
        idx += 1
    args.pos2idx = pos2idx
    print("pos2idx：", pos2idx)

    ratio = 0.3
    emb_size = 200
    layer = 1
    splitter = " "
    print("ratio=%s,emb_size=%d,layer=%d" % (str(ratio), emb_size, layer))
    # emb_path = '../ser/embeddings_%.1f_%d_%d_timelines.txt' % (ratio, emb_size, layer)
    # emb_path = 'D:/gitBase/NeuralTemporalRelation-EMNLP19/ser/embeddings_%.1f_%d_%d_timelines.txt' % (ratio, emb_size, layer)
    # mdl_path = 'D:/gitBase/NeuralTemporalRelation-EMNLP19/comm_sense/model/pairwise_model.pt'
    # mdl_path = 'D:/gitBase/NeuralTemporalRelation-EMNLP19/ser/pairwise_model_%.1f_%d_%d.pt' % (ratio, emb_size, layer)
    # mdl_path = 'D:/gitBase/NeuralTemporalRelation-EMNLP19/ser/pairwise_model_model_%.1f_%d_%d.pt' % (ratio, emb_size, layer)

    emb_path = '/usr/local/deeplearning/EventExtraction/NeuralTemporalRelation-EMNLP19/ser/embeddings_%.1f_%d_%d_timelines.txt' % (
    ratio, emb_size, layer)
    mdl_path = '/usr/local/deeplearning/EventExtraction/NeuralTemporalRelation-EMNLP19/modeldownload/ser/pairwise_model_%.1f_%d_%d.pt' % (
    ratio, emb_size, layer)

    # 初始化
    bigramGetter3 = bigramGetter_fromNN3(emb_path=emb_path, mdl_path=mdl_path, ratio=ratio, layer=layer,
                                         emb_size=emb_size, splitter=splitter)
    args.bigramGetter = bigramGetter3
    args.all_verbs_len = bigramGetter3.all_verbs_len
    main(args)
    print("BERT Done!")
