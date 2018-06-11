import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
import struct

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

# class Poetry(nn.Module):
#     def __init__(self, vocab_size, embedding_size, hidden_size):
#         super(Poetry, self).__init__()
#         self.hidden_size = hidden_size
#         self.embedding = nn.Embedding(vocab_size, embedding_size)
#         self.lstm = nn.LSTM(embedding_size, hidden_size)
#         self.l1 = nn.Linear(hidden_size, vocab_size)
#         # self.dropout = nn.Dropout(0.2)
#         self.softmax = nn.Softmax(dim=1)

#     def forward(self, inpot, hidden):
#         size = len(inpot)
#         out = self.embedding(inpot)
#         out, hide = self.lstm(out.view(size, 1, -1), hidden)
#         out = F.relu(self.l1(out.view(size, -1)))
#         # out = self.dropout(out)
#         out = self.softmax(out)
#         return out, hide

#     def init_hidden(self):
#         return (autograd.Variable(torch.zeros(1, 1, self.hidden_size, device=device)),
#                 autograd.Variable(torch.zeros(1, 1, self.hidden_size, device=device)))


# net = Poetry(5, 10, 10)

# optimizer = optim.RMSprop(net.parameters(), lr=0.01, weight_decay=0.0001)
# criterion = nn.NLLLoss()


poem = open("poem.dat", "rb")
num, = struct.unpack(">i", poem.read(4))
for _ in range(num):
    size, = struct.unpack(">i", poem.read(4))
    content = struct.unpack(">{}i".format(int(size)), poem.read(size*4))
