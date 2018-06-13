import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import struct
import random
import datetime

class Model(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.8)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inpot, hidden):
        size = len(inpot)
        out = self.embedding(inpot)
        out, hide = self.lstm(out.view(size, 1, -1), hidden)
        out = self.dropout(out)
        out = F.relu(self.l1(out.view(size, -1)))
        out = self.softmax(out)
        return out, hide


class Poetry():
    def __init__(self, key_set, poem_set, embedding_size=128, hidden_size=128):
        self.keys = key_set
        self.keys.extend(['<STA>', '<EOF>'])
        self.STA = len(self.keys)-2
        self.EOF = len(self.keys)-1
        self.poems = poem_set
        self.vocab_size = len(key_set)
        self.hidden_size = hidden_size
        self.net = Model(self.vocab_size, embedding_size, self.hidden_size)
        self.optimizer = optim.RMSprop(self.net.parameters(), lr=0.01, weight_decay=0.0001)
        self.criterion =  nn.NLLLoss()
    
    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_size)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_size)))
    
    def save(self, filename="poetry.model"):
        torch.save(self.net, filename)

    def load(self, filename="poetry.model"):
        self.net = torch.load(filename)

    def write(self, start='', max=50, force=False):
        start_index = random.randint(0, self.vocab_size)
        if start != '':
            start_index = self.keys.index(start)
        content = [self.keys[start_index]]
        hidden = self.init_hidden()
        iop = start_index
        for _ in range(max-1):
            output, hidden = self.net(torch.LongTensor([iop]), hidden)
            _, topi = output.topk(1)
            index = topi[0][0]
            content.append(self.keys[index])
            if index == self.EOF and (not force):
                break
            iop = index
        return "".join(content)

    def train_i(self, start_index, end_index):
        for index in range(start_index, end_index):
            self.net.zero_grad()
            pm = [self.STA] + self.poems[index] + [self.EOF]
            hidden = self.init_hidden()
            ip = autograd.Variable(torch.LongTensor(pm[:-1]))
            it = autograd.Variable(torch.LongTensor(pm[1:]))
            op, hidden = self.net(ip, hidden)
            loss = self.criterion(op, it)
            loss.backward()
            self.optimizer.step()
            if (index-start_index+1)%50 == 0:
                now = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                print("{} ({:5d})  {:.4f}  {}".format(
                    now, index+1, self.test(0, 10), self.write('<STA>')))

    def train(self, epoch):
        for i in range(0, epoch):
            print("==== epoch {} ====".format(i+1))
            self.train_i(0, len(self.poems))

    def test(self, start, end):
        loss = 0
        count = 0
        for c in range(start, end):
            pm = [self.STA] + self.poems[c] + [self.EOF]
            hidden = self.init_hidden()
            ip = autograd.Variable(torch.LongTensor(pm[:-1]))
            it = autograd.Variable(torch.LongTensor(pm[1:]))
            op, hidden = self.net(ip, hidden)
            loss += self.criterion(op, it)
            count += 1
        loss = loss / count
        return loss.item()


pf = open("poem.dat", "rb")
num, = struct.unpack(">I", pf.read(4))
poems = []
for _ in range(num):
    size, = struct.unpack(">H", pf.read(2))
    content = struct.unpack(">{}H".format(size), pf.read(size*2))
    poems.append(list(content))
pf.close()
kf = open("key.txt", "r", encoding="utf-8")
keys = kf.read().split("\n")
kf.close()

poet = Poetry(keys, poems)
