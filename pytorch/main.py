import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import struct

class Poetry(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size):
        super(Poetry, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.lstm = nn.LSTM(embedding_size, hidden_size)
        self.l1 = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inpot, hidden):
        size = len(inpot)
        out = self.embedding(inpot)
        out, hide = self.lstm(out.view(size, 1, -1), hidden)
        out = F.relu(self.l1(out.view(size, -1)))
        out = self.dropout(out)
        out = self.softmax(out)
        return out, hide

    def eval(self, inpot, hidden):
        size = len(inpot)
        out = self.embedding(inpot)
        out, hide = self.lstm(out.view(size, 1, -1), hidden)
        out = F.relu(self.l1(out.view(size, -1)))
        out = self.softmax(out)
        return out, hide

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_size)).cuda(),
                autograd.Variable(torch.zeros(1, 1, self.hidden_size)).cuda())


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

EOF_INDEX = len(keys)
keys.append('<EOF>')

def decode(poem):
    return "".join([keys[r] for r in poem])

net = Poetry(len(keys), 512, 512)
net.cuda()
optimizer = optim.RMSprop(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

def write(start, n):
    content = [keys[start]]
    hidden = net.init_hidden()
    iop = start
    for _ in range(n):
        output, hidden = net.eval(torch.LongTensor([iop]).cuda(), hidden)
        _, topi = output.topk(1)
        index = topi[0][0]
        content.append(keys[index])
        iop = index
    return "".join(content)

def train(f, t):
    net.zero_grad()
    loss = 0
    count = 0
    for c in range(f, t):
        pm = poems[c]
        hidden = net.init_hidden()
        ip = autograd.Variable(torch.LongTensor(pm))
        temp = pm[1:]
        temp.append(EOF_INDEX)
        ie = autograd.Variable(torch.LongTensor(temp))
        op, hidden = net(ip.cuda(), hidden)
        loss += criterion(op, ie.cuda())
        count += 1         
    loss = loss / count
    loss.backward()
    optimizer.step()
            
def test():
    loss = 0
    count = 0
    for c in range(0, 100):
        pm = poems[c]
        hidden = net.init_hidden()
        ip = autograd.Variable(torch.LongTensor(pm))
        temp = pm[1:]
        temp.append(EOF_INDEX)
        ie = autograd.Variable(torch.LongTensor(temp))
        op, hidden = net(ip.cuda(), hidden)
        loss += criterion(op, ie.cuda())
        count += 1
    loss = loss / count
    print(loss.item())
