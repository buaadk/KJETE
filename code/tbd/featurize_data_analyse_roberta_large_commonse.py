import pickle
from dataclasses import dataclass
from typing import Tuple
import argparse
from collections import defaultdict, Counter, OrderedDict
import random
import logging as log
from transformers import RobertaTokenizer, RobertaModel
from transformers import RobertaModel
import os
import torch
from torch.utils import data
import time, copy
# from featureFuncs import *
from featureFuncsExec import *
from sklearn.model_selection import train_test_split
from eventpairwise_ffnn_pytorch import VerbNet
from mySysLstm import lstm_siam

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


class Event():
    id: str
    type: str
    text: str
    tense: str
    polarity: str
    span: (int, int)


def check_getback_loc(orgSent, robertaSent):
    robertaSent.pop(0)
    robertaSent.pop(-1)
    sentList = []
    while len(robertaSent) > 0 and len(orgSent) > 0:
        popWord = orgSent.pop(0)
        cmpWord = robertaSent.pop(0)
        num = 1
        while popWord != cmpWord:
            cmpWord += robertaSent.pop(0)
            num += 1
        sentList.append((popWord, num))
    return sentList


def transform2postion(sentList, orgSent):
    # sentList.append(("</s>",1))
    orig_to_tok_List = []
    # orig_to_tok_List.append(("<s>",0))
    pointer = 1
    for i, (word, num) in enumerate(sentList):
        orig_to_tok_List.append((word, pointer))
        pointer += num

    correspondingList = []
    for i, word in enumerate(orgSent):
        start = i
        word2, start_s = orig_to_tok_List[start]
        word3, num = sentList[start]
        assert word == word2
        assert word == word3
        end_s = start_s + num - 1
        correspondingList.append((word, start_s, end_s))
    return correspondingList


def create_features(ex, pos2idx, tokenizer, roberta_model, lstmsiammodel):
    # bert_model.eval()
    pos_dict = ex['doc_dictionary']
    ent_labels = ex['event_labels']

    all_keys, lidx_start, lidx_end, ridx_start, ridx_end = \
        token_idx(ex['left_event'].span, ex['right_event'].span, pos_dict)

    # truncate dictionary into three pieces
    left_seq = [pos_dict[x][0] for x in all_keys[:lidx_start]]
    right_seq = [pos_dict[x][0] for x in all_keys[ridx_end + 1:]]
    in_seq = [pos_dict[x][0] for x in all_keys[lidx_start:ridx_end + 1]]

    # find context sentence(s) start and indices
    try:
        sent_start = max(loc for loc, val in enumerate(left_seq) if val == ".") + 1
    except:
        sent_start = 0

    try:
        sent_end = ridx_end + 1 + min(loc for loc, val in enumerate(right_seq) if val == '.')
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
    org_sent = [pos_dict[x][0].lower() for x in sent_key]
    # sent = [args.w2i[t] if t in args.w2i.keys() else 1 for t in org_sent]

    pos = [pos2idx[k] if k in pos2idx.keys() else len(pos2idx) for k in [pos_dict[x][1] for x in sent_key]]
    ent = [(x, ent_labels[x]) for x in sent_key]

    # calculate events' index in context sentences
    lidx_start_s = lidx_start - sent_start
    lidx_end_s = lidx_end - sent_start
    ridx_start_s = ridx_start - sent_start
    ridx_end_s = ridx_end - sent_start

    # bert sentence segment ids
    seg = 0
    bert_pos = []
    bert_ent = []

    # append sentence start
    # original token to bert word-piece token mapping
    orig_to_tok_map = []

    # bert_pos.append("[CLS]")
    bert_pos.append("<s>")

    # sent_start is non-event by default
    # bert_ent.append(("[CLS]", 0))
    bert_ent.append(("<s>", 0))

    # roberta_words = []
    # roberta_words_Id = []
    content = " ".join(org_sent)

    lefteventname, righteventname = ex['left_event'].text, ex['right_event'].text

    roberta_subwords = []
    encoded = tokenizer.encode(content)

    for index, i in enumerate(encoded):
        r_token = tokenizer.decode([i])
        r_token = r_token.strip()
        roberta_subwords.append(r_token)

    cporgsent = copy.deepcopy(org_sent)
    cproberta_subwords = copy.deepcopy(roberta_subwords)

    sentList = check_getback_loc(cporgsent, cproberta_subwords)
    correspondingList = transform2postion(sentList, org_sent)

    # roberta_words_Id.append(encoded)
    for index, subword in enumerate(org_sent):
        (word, subLen) = sentList[index]
        for _ in range(0, subLen):
            bert_pos.append(pos[index])
            bert_ent.append(ent[index])

    # print("org_sent:", " ".join(org_sent))
    # print("roberta_subwords:", " ".join(roberta_subwords))

    bert_pos.append("</s>")
    bert_ent.append(("</s>", 0))

    # map original token index into bert (word_piece) index
    word1, lidx_start_s, lidx_end_s = correspondingList[lidx_start_s]
    left_event = ex['left_event'].text
    left_event = left_event.lower()
    # print("left_event:",left_event,"word1:",word1)
    # assert left_event == word1
    # lidx_start_s = correspondingList[lidx_start_s]
    # lidx_end_s = orig_to_tok_map[lidx_end_s + 1] - 1

    word2, ridx_start_s, ridx_end_s = correspondingList[ridx_start_s]
    right_event = ex['right_event'].text
    right_event = right_event.lower()
    # print("right_event:", right_event, "word2:", word2)
    # assert right_event == word2
    # ridx_start_s = orig_to_tok_map[ridx_start_s]
    # ridx_end_s = orig_to_tok_map[ridx_end_s + 1] - 1

    # bert_sent = tokenizer.convert_tokens_to_ids(bert_tokens)

    # bert_sent = torch.tensor([bert_sent])
    bert_sent = torch.tensor([encoded])

    # use the last layer computed by BERT as token vectors
    try:
        out = roberta_model(bert_sent)[0]
        sent = out[-1].squeeze(0).data.numpy()
    except:
        sent_len = len(content)
        print("error:", sent_len, pred_ind)
        sent = []
        bert_pos = []

    # create lexical features for the model
    # new_fts = []
    # new_fts.append(-distance_features(lidx_start, lidx_end, ridx_start, ridx_end))
    distanceMessage = -distance_features(lidx_start, lidx_end, ridx_start, ridx_end)
    # new_fts.append(distanceMessage)
    fts = []
    fts.append(distanceMessage)
    fts = np.array(fts)
    dismsg = torch.from_numpy(fts).unsqueeze(1)

    word1 = ex['left_event'].text
    word2 = ex['right_event'].text
    commonSense = lstmsiammodel(word1, word2)
    ftsMsg = torch.cat([dismsg, commonSense], dim=1)

    # print(lidx_start_s, lidx_end_s)
    return (sent, bert_ent, bert_pos, ftsMsg, ex['rev'], lidx_start_s, lidx_end_s, ridx_start_s, ridx_end_s, pred_ind)


def parallel(ex, ex_id, args, tokenizer, bert_model, lstmsiammodel):
    label_id = args._label_to_id[ex['rel_type']]

    return ex['doc_id'], ex_id, (ex['left_event'].id, ex['right_event'].id), label_id, \
           create_features(ex, args.pos2idx, tokenizer, bert_model, lstmsiammodel)


def data_split(train_docs, eval_docs, data, neg_r=0.0, seed=7):
    train_set = []
    eval_set = []
    train_set_neg = []

    for s in data:
        # dev-set doesn't require unlabeled data
        if s[0] in eval_docs:
            # 0:doc_id,1:ex.id,2:(ex.left.id,ex.right.id),3:label_id,4:features
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


def tokenized_to_origin_span(sent, token_list):
    token_span = []
    pointer = 0
    for token in token_list:
        while True:
            if token[0] == sent[pointer]:
                start = pointer
                end = start + len(token) - 1
                pointer = end + 1
                break
            else:
                pointer += 1
        token_span.append([start, end])
    return token_span


def split_and_save(train_docs, dev_docs, data, seed, save_dir, nr=0.0):
    # first split labeled into train and dev
    train_data, dev_data = data_split(train_docs, dev_docs, data, neg_r=nr)
    print("len(train_data):", len(train_data), "len(dev_data):", len(dev_data))

    # shuffle
    # random.Random(seed)
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    with open(save_dir + "/train.pickle", "wb") as handle:
        pickle.dump(train_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

    with open(save_dir + "/dev.pickle", "wb") as handle:
        pickle.dump(dev_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()

    return


def exec_func(args):
    # to pick up here
    if args.data_type == "matres":
        label_map = matres_label_map
    elif args.data_type == "tbd":
        label_map = tbd_label_map
    elif args.data_type == "comsense":
        label_map = tbd_label_map

    all_labels = list(OrderedDict.fromkeys(label_map.values()))

    args._label_to_id = OrderedDict([(all_labels[l], l) for l in range(len(all_labels))])
    args._id_to_label = OrderedDict([l, all_labels[l]] for l in range(len(all_labels)))

    print(args._label_to_id)
    print(args._id_to_label)

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #tokenizer = RobertaTokenizer.from_pretrained('roberta-base', use_token='<unk>')
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large', use_token='<unk>')
    # bert_model = BertModel.from_pretrained('../bert-base-uncased/')
    #robertaModel = RobertaModel.from_pretrained('roberta-base')
    robertaModel = RobertaModel.from_pretrained('roberta-large')

    train_data = pickle.load(open(args.data_dir + "/train.pickle", "rb"))

    bigramGetterTemp = args.bigramGetter
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
    lstmsiammodel = lstm_siam(params, bigramGetter=bigramGetterTemp, granularity=0.1,
                              common_sense_emb_dim=64,
                              bidirectional=True, lowerCase=False, verb_i_map=bigramGetterTemp.verb_i_map)

    print("process train ...")

    data = [parallel(v, k, args, tokenizer, robertaModel, lstmsiammodel) for k, v in train_data.items()]

    # doc splits
    if args.data_type in ['tbd', 'comsense']:
        print("process dev...")
        dev_data = pickle.load(open(args.data_dir + "/dev.pickle", "rb"))
        dev_data = [parallel(v, k, args, tokenizer, robertaModel, lstmsiammodel) for k, v in dev_data.items()]
        data += dev_data

    # doc splits
    if args.data_type in ['matres']:
        train_docs, dev_docs = train_test_split(args.train_docs, test_size=0.2, random_state=args.seed)
    # TBDense data has given splits on train/dev/test
    else:
        train_docs = args.train_docs
        dev_docs = args.dev_docs

    # save_data_dir
    if not os.path.isdir(args.save_data_dir):
        os.mkdir(args.save_data_dir)

    if 'all' in args.split:
        print("process test...")
        test_data = pickle.load(open(args.data_dir + '/test.pickle', "rb"))
        test_data = [parallel(v, k, args, tokenizer, robertaModel, lstmsiammodel) for k, v in test_data.items()]

        with open(args.save_data_dir + '/test.pickle', "wb") as handle:
            pickle.dump(test_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.close()

        split_and_save(train_docs, dev_docs, data, args.seed, args.save_data_dir)
        # quick trick to reduce number of tokens in GloVe

    return


class bigramGetter_fromNN_Temprob:
    def __init__(self, emb_path, mdl_path, ratio=0.3, layer=1, emb_size=200, splitter=','):
        self.verb_i_map = {}
        f = open(emb_path)
        lines = f.readlines()
        for i, line in enumerate(lines):
            msg = line.split(splitter)
            msg = msg[0]
            self.verb_i_map[msg] = i
        f.close()

        # hidden_ratio = 128
        # emb_size = 128
        # all_verbs_len = 487
        self.all_verbs_len = len(self.verb_i_map)
        self.model = VerbNet(self.all_verbs_len, hidden_ratio=ratio, emb_size=emb_size, num_layers=layer)
        checkpoint = torch.load(mdl_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model_state_dict'])

    def eval(self, v1, v2):
        return self.model(torch.from_numpy(np.array([[self.verb_i_map[v1], self.verb_i_map[v2]]])))

    def getBigramStatsFromTemprob(self, word1, word2):
        v1, v2 = word1, word2
        if v1 not in self.verb_i_map or v2 not in self.verb_i_map:
            return torch.FloatTensor([0, 0]).view(1, -1)
        return torch.cat((self.eval(v1, v2), self.eval(v2, v1)), 1).view(1, -1)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-data_dir', type=str, default='../data/')
    p.add_argument('-other_dir', type=str, default='../other/')
    p.add_argument('-load_model_dir', type=str, default='')
    p.add_argument('-train_docs', type=list, default=[])
    p.add_argument('-dev_docs', type=list, default=[])
    p.add_argument('-split', type=str, default='bert_all_joint_cosmos')
    p.add_argument('-data_type', type=str, default='comsense')
    #p.add_argument('-data_type', type=str, default='matres')
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
    # print(args.train_docs[:10])
    args.save_data_dir = args.data_dir + '/' + args.split

    # glove = read_glove(args.other_dir + "/glove.6B.50d.txt")
    # glove = read_glove(args.other_dir + "/glove.6B.50d2.txt")
    # vocab = np.array(['<pad>', '<unk>'] + list(glove.keys()))
    # args.w2i = OrderedDict((vocab[i], i) for i in range(len(vocab)))

    tags = open(args.other_dir + "/pos_tags.txt")
    pos2idx = {}
    idx = 0
    for tag in tags:
        tag = tag.strip()
        pos2idx[tag] = idx
        idx += 1
    args.pos2idx = pos2idx

    ratio = 0.3
    emb_size = 200
    layer = 1
    splitter = " "
    print("ratio=%s,emb_size=%d,layer=%d" % (str(ratio), emb_size, layer))
    # emb_path = 'D:/gitBase/NeuralTemporalRelation-EMNLP19/ser/embeddings_%.1f_%d_%d_timelines.txt' % (ratio, emb_size, layer)
    # mdl_path = 'D:/gitBase/NeuralTemporalRelation-EMNLP19/ser/pairwise_model_%.1f_%d_%d.pt' % (ratio, emb_size, layer)

    emb_path = '/usr/local/deeplearning/EventExtraction/NeuralTemporalRelation-EMNLP19/ser/embeddings_%.1f_%d_%d_timelines.txt' % (
    ratio, emb_size, layer)
    mdl_path = '/usr/local/deeplearning/EventExtraction/NeuralTemporalRelation-EMNLP19/modeldownload/ser/pairwise_model_%.1f_%d_%d.pt' % (
    ratio, emb_size, layer)

    # init
    bigramGetterTemp = bigramGetter_fromNN_Temprob(emb_path=emb_path, mdl_path=mdl_path, ratio=ratio, layer=layer,
                                                   emb_size=emb_size, splitter=splitter)
    args.bigramGetter = bigramGetterTemp
    args.all_verbs_len = bigramGetterTemp.all_verbs_len

    print(emb_path)

    exec_func(args)
    print("########")
