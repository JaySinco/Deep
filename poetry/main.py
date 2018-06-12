import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import struct
import random
import datetime

class Poetry(nn.Module):
    def __init__(self, key_set, poem_set, embedding_size=512, hidden_size=512):
        super(Poetry, self).__init__()
        self.keys = key_set
        self.poems = poem_set
        self.EOF = len(key_set)
        self.vocab_size = len(key_set) + 1
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(self.vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, self.vocab_size)
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.LogSoftmax(dim=1)
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.01, weight_decay=0.0001)
        self.criterion =  nn.NLLLoss()

    def forward(self, inpot, hidden):
        size = len(inpot)
        out = self.embedding(inpot)
        out, hide = self.lstm(out.view(size, 1, -1), hidden)
        out = self.dropout(out)
        out = F.relu(self.l1(out.view(size, -1)))
        out = self.softmax(out)
        return out, hide

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_size)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_size)))
    
    def write(self, start='', max=50):
        start_index = random.randint(0, self.vocab_size-2)
        if start != '':
            start_index = self.keys.index(start)
        content = [self.keys[start_index]]
        hidden = self.init_hidden()
        iop = start_index
        for _ in range(max-1):
            output, hidden = self.forward(torch.LongTensor([iop]), hidden)
            _, topi = output.topk(1)
            index = topi[0][0]
            if index == self.EOF:
                break
            content.append(self.keys[index])
            iop = index
        return "".join(content)

    def train_i(self, start_index, end_index):
        for index in range(start_index, end_index):
            if (index-start_index)%50 == 0:
                now = datetime.datetime.now().strftime('%Y/%m/%d %H:%M:%S')
                print("{} ({:5d})  {}".format(now, index, self.test(0, 50)))
            self.zero_grad()
            pm = self.poems[index]
            hidden = self.init_hidden()
            ip = autograd.Variable(torch.LongTensor(pm))
            temp = pm[1:]
            temp.append(self.EOF)
            it = autograd.Variable(torch.LongTensor(temp))
            op, hidden = self.forward(ip, hidden)
            loss = self.criterion(op, it)
            loss.backward()
            self.optimizer.step()

    def train(self, epoch=1):
        for i in range(0, epoch):
            print("==== epoch {} ====".format(i))
            self.train_i(0, len(self.poems))

    def test(self, start, end):
        loss = 0
        count = 0
        for c in range(start, end):
            pm = poems[c]
            hidden = self.init_hidden()
            ip = autograd.Variable(torch.LongTensor(pm))
            temp = pm[1:]
            temp.append(self.EOF)
            it = autograd.Variable(torch.LongTensor(temp))
            op, hidden = self.forward(ip, hidden)
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

net = Poetry(keys, poems)
net.train(30)
torch.save(net, "poetry.torch")
