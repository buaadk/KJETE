import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, recall_score, precision_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, classification_report, accuracy_score, f1_score, \
    confusion_matrix
import re
import sys
import json
import numpy as np



def CM_metric(CM):
    all_ = CM.sum()

    Acc = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2] + CM[3][3]) / all_
    P = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2]) / (CM[0][0:3].sum() + CM[1][0:3].sum() + CM[2][0:3].sum() + CM[3][0:3].sum())
    R = 1.0 * (CM[0][0] + CM[1][1] + CM[2][2]) / (CM[0].sum() + CM[1].sum() + CM[2].sum())
    F1 = 2 * P * R / (P + R)

    return Acc, P, R, F1, CM

def metric(y_true, y_pred):
    CM = confusion_matrix(y_true, y_pred)
    Acc, P, R, F1, _ = CM_metric(CM)

    return Acc, P, R, F1, CM

# Usage :python pairwise_ffnn_pytorch.py hidden_ration emb_dim num_layers training_set

class VerbNet(nn.Module):
    def __init__(self, vocab_size, hidden_ration=0.5, emb_size=128, num_layers=1):
        super(VerbNet, self).__init__()
        self.emb_size = emb_size
        self.emb_layer = nn.Embedding(vocab_size, self.emb_size)
        self.fc1 = nn.Linear(self.emb_size * 2, int(self.emb_size * 2 * hidden_ration))
        self.num_layers = num_layers
        if num_layers == 1:
            self.fc2 = nn.Linear(int(self.emb_size * 2 * hidden_ration), 1)
        else:
            self.fc2 = nn.Linear(int(self.emb_size * 2 * hidden_ration), int(self.emb_size * hidden_ration))
            self.fc3 = nn.Linear(int(self.emb_size * hidden_ration), 1)
        self.is_training = True

    def forward(self, x):
        x_emb = self.emb_layer(x)
        dim0 = x_emb[:, 0, :]
        dim1 = x_emb[:, 1, :]
        fullX = torch.cat((x_emb[:, 0, :], x_emb[:, 1, :]), dim=1)
        layer1 = F.relu(self.fc1(F.dropout(fullX, p=0.3, training=self.is_training)))
        if self.num_layers == 1:
            return torch.sigmoid(self.fc2(layer1))
        layer2 = F.relu(self.fc2(F.dropout(layer1, p=0.3, training=self.is_training)))
        layer3 = torch.sigmoid(self.fc3(layer2))
        return layer3

    def retrieveEmbeddings(self, x):
        x_emb = self.emb_layer(x)
        fullX = torch.cat((x_emb[:, 0, :], x_emb[:, 1, :]), dim=1)
        layer1 = F.relu(self.fc1(fullX))
        if self.num_layers == 1:
            return layer1
        layer2 = F.relu(self.fc2(layer1))
        return torch.cat((layer1, layer2), 1)


class FfnnTrainer():
    def __init__(self, ffnn, batch_size=64):
        self.ffnn = ffnn
        self.optimizer = torch.optim.Adam(ffnn.parameters(), lr=1e-4)
        self.loss = nn.BCELoss()
        #self.suffix = '_' + sys.argv[1] + '_' + sys.argv[2] + '_' + sys.argv[3] + '_' + sys.argv[4]
        self.batch_size = batch_size

    def train(self, X_train, Y_train, counts_train, X_test, Y_test, counts_test):
        loss_value = np.inf
        prev_loss_value = np.inf
        count = 1
        batch_size = self.batch_size
        train_losses = []
        test_recalls = []
        test_precisions = []
        test_losses = []

        while count <= 5000:
            prev_loss_value = loss_value
            loss_value = 0
            self.ffnn.is_training = True
            start = time.time()
            for i in range(0, X_train.shape[0], batch_size):
                x = np.int64(X_train[i:min(i + batch_size, X_train.shape[0]), :])
                c = counts_train[i:min(i + batch_size, X_train.shape[0])]
                x = np.repeat(x, c, axis=0)
                y = Y_train[i:min(i + batch_size, X_train.shape[0])]
                y = np.repeat(y, c, axis=0)
                self.optimizer.zero_grad()
                y_pred = self.ffnn(torch.from_numpy(x))
                L = self.loss(y_pred.float(), torch.from_numpy(y).float())
                loss_value += L.item()
                L.backward()
                self.optimizer.step()
            end = time.time()
            print(count, loss_value, 'time', end - start)
            train_losses.append(loss_value)
            if count % 2 == 0:
                self.ffnn.is_training = False
                y_true = []
                y_pred = []
                start = time.time()
                loss_value = 0
                for i in range(0, X_test.shape[0], batch_size):
                    x = np.int64(X_test[i:min(i + batch_size, X_test.shape[0]), :])
                    c = counts_test[i:min(i + batch_size, X_test.shape[0])]
                    x = np.repeat(x, c, axis=0)
                    y = Y_test[i:min(i + batch_size, X_test.shape[0])]
                    y = np.repeat(y, c, axis=0)
                    y_true += list(np.int32(y >= 0.5))
                    y_pre = self.ffnn(torch.from_numpy(x)).cpu().detach().numpy()
                    y_pred += list(np.int32(y_pre >= 0.5))
                recall = recall_score(y_true, y_pred, average='micro')
                precision = precision_score(y_true, y_pred, average='micro')
                f1 = f1_score(y_true, y_pred, average='micro')
                test_recalls.append(recall)
                test_precisions.append(precision)
                if count % 5 == 0:
                    torch.save(self.ffnn.state_dict(), './pairwise_model.pt')
                end = time.time()
                print(count, precision, recall, f1, 'time', end - start)

            count += 1


if __name__ == '__main__':
    all_verbs = set()
    pair_map = {}
    total_pairs = 0
    temprob = open('../data/temprob.txt')
    #temprob = open('/usr/local/deeplearning/EventExtraction/NeuralTemporalRelation-EMNLP19/temporalRelation/ser/ser/temprob.txt')
    lines = temprob.readlines()
    for i, line in enumerate(lines):
        period_count = 0
        parts = line.split()
        for c in parts[0]:
            if c == '.':
                period_count += 1
        if period_count > 1:
            continue
        period_count = 0
        for c in parts[1]:
            if c == '.':
                period_count += 1
        if period_count > 1:
            continue
        part1parts = parts[0].split('.')
        word1 = part1parts[0]
        parts2parts = parts[1].split('.')
        word2 = parts2parts[0]
        all_verbs.add(word1)
        all_verbs.add(word2)
        first = word1
        second = word2
        relation = parts[2]
        if relation not in {'before', 'after'}:
            continue
        if first not in pair_map:
            pair_map[first] = {}
        if second not in pair_map[first]:
            total_pairs += 1
            pair_map[first][second] = {'after': 0, 'before': 0}
        pair_map[first][second][relation] += int(parts[3])
        #if i % 100 == 0:
        #    print(i)

            # print(parts)

    all_verbs = sorted(list(all_verbs))
    print("len", len(all_verbs))
    verb_i_map = {}
    X = np.zeros((total_pairs, 2))
    Y = np.zeros((total_pairs, 1))
    counts = np.zeros((total_pairs), dtype=np.int64)
    for i, verb in enumerate(all_verbs):
        verb_i_map[verb] = i
    index = 0
    for verb1 in pair_map:
        for verb2 in pair_map[verb1]:
            X[index, 0] = verb_i_map[verb1]
            X[index, 1] = verb_i_map[verb2]
            counts[index] = pair_map[verb1][verb2]['before'] + pair_map[verb1][verb2]['after']
            Y[index] = float(pair_map[verb1][verb2]['before']) / counts[index]
            index += 1
    print('train test split')
    hidden_ratio = 128
    emb_size = 128
    num_layers = 1
    X_train, X_test, Y_train, Y_test, counts_train, counts_test = train_test_split(X, Y, counts, test_size=0.2)
    print("Train len", len(X_train), "X_test len", len(X_test))
    # the paper model
    ffnn = VerbNet(len(all_verbs), hidden_ratio, emb_size, num_layers)
    batch_size = 32
    trainer = FfnnTrainer(ffnn, batch_size=batch_size)
    trainer.train(X_train, Y_train, counts_train, X_test, Y_test, counts_test)
    ffnn.is_training = False
    y_true = []
    y_pred = []
    for i in range(0, X_test.shape[0], batch_size):
        x = np.int64(X_test[i:min(i + batch_size, X_test.shape[0]), :])
        c = counts_test[i:min(i + batch_size, X_test.shape[0])]
        x = np.repeat(x, c, axis=0)
        y = Y_test[i:min(i + batch_size, X_test.shape[0])]
        y = np.repeat(y, c, axis=0)
        y_true += list(np.int32(y >= 0.5))
        y_pre = ffnn(torch.from_numpy(x))
        y_pred += list(np.int32(y_pre >= 0.5))
    recall = recall_score(y_true, y_pred, average='micro')
    precision = precision_score(y_true, y_pred, average='micro')
    f1 = f1_score(y_true, y_pred, average='micro')
    print('final precision:', precision, 'recall:', recall, 'f1:', f1)
