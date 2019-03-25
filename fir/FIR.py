import os
import sys
import re
import time
import math
import heapq
import logging
import random
import mxnet as mx

logging.basicConfig(level=logging.INFO,  
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',  
                    datefmt='%Y/%m/%d %H:%M:%S') 

EXPLORE_RATE = 1
LEARNING_RATE = 0.1
MCTS_ITER_MAX = 100
SEC_PER_EPOCH = 90
BACKUP_LAST_SEC = 60 * 15
EPOCH_PER_BACKUP = int(BACKUP_LAST_SEC / SEC_PER_EPOCH)

BOARD_ROW = 8
BOARD_COL = 8
BOARD_DIM = (BOARD_ROW, BOARD_COL)
BOARD_SIZE = BOARD_ROW * BOARD_COL
INPUT_FEATHURE_DIM = (2, *BOARD_DIM)

logging.info("backup network parameters every {} epochs".format(EPOCH_PER_BACKUP))

class Residual(mx.gluon.nn.Block):
    def __init__(self, num_channels, **kwargs):
        super(Residual, self).__init__(**kwargs)
        self.conv1 = mx.gluon.nn.Conv2D(num_channels, kernel_size=3, padding=1)
        self.bn1 = mx.gluon.nn.BatchNorm()
        self.conv2 = mx.gluon.nn.Conv2D(num_channels, kernel_size=3, padding=1)
        self.bn2 = mx.gluon.nn.BatchNorm()

    def forward(self, X):
        Y = mx.nd.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        return mx.nd.relu(Y+X)

def resnet_block(num_channels, num_residuals):
    blk = mx.gluon.nn.Sequential()
    for i in range(num_residuals):
        blk.add(Residual(num_channels))
    return blk

def conv_block(num_channels, kernel_size, padding):
    blk = mx.gluon.nn.Sequential()
    blk.add(mx.gluon.nn.Conv2D(num_channels, kernel_size=kernel_size, padding=padding))
    blk.add(mx.gluon.nn.BatchNorm())
    blk.add(mx.gluon.nn.Activation("relu"))
    return blk

class PVNet(mx.gluon.nn.Block):
    def __init__(self, id, verno=None, num_channels=64, **kwargs):
        super(PVNet, self).__init__(**kwargs)
        self.core = mx.gluon.nn.Sequential()
        self.core.add(conv_block(num_channels, 3, 1))
        self.core.add(resnet_block(num_channels, 5))
        self.plc = mx.gluon.nn.Sequential()
        self.plc.add(conv_block(2, 1, 0))
        self.plc.add(mx.gluon.nn.Dense(BOARD_SIZE))
        self.val = mx.gluon.nn.Sequential()
        self.val.add(conv_block(1, 1, 0))
        self.val.add(mx.gluon.nn.Dense(num_channels, activation='relu'))
        self.val.add(mx.gluon.nn.Dense(1))
        self.id = id
        if verno is None:
            self.verno = self.restore()
        elif verno == 0:
            self.initialize()
            self.verno = 0
        else:
            self.verno = self.restore(verno)
        logging.info('{}.verno={}'.format(self.id, self.verno))

    def backup(self):
        self.save_parameters('FIR{}_{}.param'.format(self.verno, time.strftime('%y-%m-%d_%H%M')))

    def restore(self, verno=None):
        verion_list, file_list = [], []
        for fn in os.listdir(os.getcwd()):
            obj = re.match(r'FIR(\d+?)_\d\d-\d\d-\d\d_\d\d\d\d\.param', fn)
            if obj:
                n = int(obj.group(1))
                verion_list.append(n)
                file_list.append(fn)
        if len(verion_list) == 0 or (verno is not None and verion_list.count(verno) == 0):
            raise Exception("failed to locate parameter file!")
        if verno is None:
            verno = max(verion_list)
        target = file_list[verion_list.index(verno)]
        logging.info('loading {}...'.format(target))
        self.load_parameters(target)
        return verno

    def forward(self, X):
        out = self.core(X)
        plc = self.plc(out)
        val = mx.nd.tanh(self.val(out))
        return plc, val

class MCTSNode:
    def __init__(self, parent, prior, move):
        self.parent = parent
        self.prior = prior
        self.move = move
        self.childrens = []
        self.visited = 0
        self.value = 0
        self.rank = 0

    def __lt__(self, other):
        return self.rank > other.rank

    def __repr__(self):
        return "children: {}/{}, prior: {:.4f}, visited: {}, value: {:.4f}".format(
            len(self.childrens), self.totalSubNode(), self.prior, self.visited, self.value)
    
    def totalSubNode(self):
        total = 0
        for child in self.childrens:
            total += 1 + child.totalSubNode()
        return total

    def calPUCTRank(self):
        if self.parent is None:
            return
        q = 0 if self.visited == 0 else self.value/self.visited
        u = EXPLORE_RATE * self.prior * math.sqrt(self.parent.visited)/(1 + self.visited)
        self.rank = q + u

    def selectChild(self):
        heapq.heapify(self.childrens)
        return self.childrens[0]

    def expandAndBackward(self, net, state):
        priors, value = net(state.board.reshape(1, *INPUT_FEATHURE_DIM))
        priors = mx.nd.softmax(priors).reshape(*BOARD_DIM)
        value = -1 * value.reshape(1).asscalar().item()
        for move in state.optionMoves:     
            prior = priors[move[0]][move[1]].asscalar().item()
            child = MCTSNode(self, prior, move)
            child.calPUCTRank()
            self.childrens.append(child)
        node = self
        while node != None:
            node.visited += 1
            node.value += value
            node.calPUCTRank()
            value *= -1
            node = node.parent

    def prun(self, occured_move):
        for child in self.childrens:
            if child.move == occured_move:
                child.parent = None
                child.move = None
                return child
        raise Exception("failed to find enemy move in child node!")

    def PUCTPredict(self):
        bestMove = None
        mostVisited = 0
        probs = mx.nd.zeros(BOARD_DIM)
        for child in self.childrens:
            probs[child.move[0]][child.move[1]] = child.visited
            if child.visited > mostVisited:
                mostVisited = child.visited
                bestMove = child.move
        probs = probs / self.visited
        return probs, bestMove

    def PUCTIter(self, net, beginState, itermax):
        for i in range(itermax):
            node = self
            state = beginState.copy()
            while len(node.childrens) > 0:
                node = node.selectChild()
                state.next(node.move[0], node.move[1])
            node.expandAndBackward(net, state)  

class Symbol:
    ColorMap = {0: ' ', 1: '●', 2: '○'}
    def __init__(self, n):
        self.value = n
    
    def __repr__(self):
        return Symbol.ColorMap[self.value]

    def flip(self):
        return self.flipMap[self]

Empty = Symbol(0)
Black = Symbol(1)
White = Symbol(2)
Symbol.flipMap = {Black: White, White: Black}
OwnSide = 0
EnemySide = 1

def fmtBoard(board):
    buf = []
    for r in range(BOARD_ROW):
        buf.append("{:2d}|".format(r%10))
        for c in range(BOARD_COL):           
            if self.board[OwnSide][row][col] == 1:
                piece = 'o'
            elif self.board[EnemySide][row][col] == 1:
                piece = 'x'
            else:
                piece = ' '
            buf.append("{}|".format(piece))
        buf.append("\n")
    buf.append("  ")
    for c in range(BOARD_COL):
        buf.append("{:-2d}".format(c%10))
    return "".join(buf)

class State:
    def __init__(self, gen_opt_mvs=True):
        self.board = mx.nd.zeros(shape=INPUT_FEATHURE_DIM)
        self.nowPlayer = Black
        self.lastMove = None
        self.optionMoves =None
        if gen_opt_mvs:
            self.optionMoves = [(r, c) for r in range(BOARD_ROW) for c in range(BOARD_COL)]
            random.shuffle(self.optionMoves)

    def __repr__(self):
        buf = []
        for r in range(BOARD_ROW):
            buf.append("{:2d}|".format(r%10))
            for c in range(BOARD_COL):
                buf.append("{}|".format(self.get(r, c)))
            buf.append("\n")
        buf.append("  ")
        for c in range(BOARD_COL):
            buf.append("{:-2d}".format(c%10))
        buf.append("\n")
        buf.append("last move: {}{}".format(self.nowPlayer.flip(), self.lastMove))
        return "".join(buf)

    def eval(self, net):
        _, val = net(self.board.reshape(1, *INPUT_FEATHURE_DIM))
        return val.reshape(1).asscalar().item()
 
    def copy(self):
        cloned = State(gen_opt_mvs=False)
        self.board.copyto(cloned.board)
        cloned.nowPlayer = self.nowPlayer
        cloned.lastMove = self.lastMove
        cloned.optionMoves = self.optionMoves.copy()
        return cloned
       
    def get(self, row, col):
        if self.board[OwnSide][row][col] == 1:
            return self.nowPlayer
        if self.board[EnemySide][row][col] == 1:
            return self.nowPlayer.flip()
        return Empty

    def next(self, row, col):
        self.board[OwnSide][row][col] = 1
        self.board = self.board.flip(axis=0)
        self.nowPlayer = self.nowPlayer.flip()
        self.lastMove = (row, col)
        self.optionMoves.remove(self.lastMove)

    def isOnBoard(self, row, col):
        return row < BOARD_ROW and row >= 0 and col < BOARD_COL and col >= 0

    def isValidMove(self, row, col):
        return self.isOnBoard(row, col) and (self.board[OwnSide][row][col] + 
                                        self.board[EnemySide][row][col] == 0)

    def lastPlayerWin(self):
        if self.lastMove is None:
            return False
        for dr, dc in [[0, 1], [1, 0], [-1, 1], [1, 1]]:
            total = 0
            for m in [1, -1]:
                r, c = self.lastMove[0], self.lastMove[1]
                while self.isOnBoard(r, c) and self.board[EnemySide][r][c] == 1:
                    total += 1
                    r += dr * m
                    c += dc * m
            if total - 1 >= 5:
                return True
        return False

class RandPlayer:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name

    def reset(self):
        pass

    def play(self, state):
        return state.optionMoves[0]

class HumanPlayer:
    def __init__(self, name):
        self._name = name

    def name(self):
        return self._name

    def reset(self):
        pass

    def play(self, state):
        while True:
            try:
                pos = input("{}(row, col)> ".format(state.nowPlayer))
                row, col = pos.split(",")
                return (int(row), int(col))
            except ValueError:
                if "quit" in pos:
                    sys.exit()  
                continue

class MCTSPlayer:
    def __init__(self, net, itermax=MCTS_ITER_MAX):
        self._name = net.id
        self.net = net       
        self.node = MCTSNode(None, 0, None)
        self.itermax = itermax

    def reset(self):
        self.node = MCTSNode(None, 0, None)

    def name(self):
        return self._name

    def play(self, state):
        if state.lastMove != None and len(self.node.childrens) > 0:
            self.node = self.node.prun(state.lastMove)
        self.node.PUCTIter(self.net, state, self.itermax)
        _, move = self.node.PUCTPredict()
        self.node = self.node.prun(move)
        return move

def train(net, itermax=MCTS_ITER_MAX, learning_rate=LEARNING_RATE):
    epoch = net.verno
    smcEntropyLoss = mx.gluon.loss.SoftmaxCrossEntropyLoss(sparse_label=False)
    l2Loss = mx.gluon.loss.L2Loss()
    trainer = mx.gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': learning_rate})
    while True:
        epoch += 1
        net.verno += 1
        game = State()
        turn = 0
        node = MCTSNode(None, 0, None)
        ind = -1
        Xs, Ps, Vs = None, None, []
        while not (game.lastPlayerWin() or len(game.optionMoves) == 0):
            node.PUCTIter(net, game, itermax)
            MCTSProbs, move = node.PUCTPredict()
            x = game.board.reshape(1, *INPUT_FEATHURE_DIM)
            Xs = x.copy() if Xs is None else mx.nd.concat(Xs, x, dim=0)
            p = MCTSProbs.reshape(1, BOARD_SIZE)
            Ps = p.copy() if Ps is None else mx.nd.concat(Ps, p, dim=0)
            ind *= -1
            Vs.append(ind)
            node = node.prun(move)
            game.next(move[0], move[1])
            turn += 1  
        if game.lastPlayerWin():
            reward = 1 if Vs[-1] == 1 else -1
            Vs = mx.nd.array(Vs) * reward
        else:
            reward = 0.5
            Vs = mx.nd.array(Vs).abs() * reward
        Vs = Vs.reshape(-1, 1)
        with mx.autograd.record():
            plc, val = net(Xs)
            vloss = l2Loss(val, Vs)
            ploss = smcEntropyLoss(plc, Ps)
            loss = (vloss + ploss).mean()
            logging.info("loss[V/P]={:.4f}/{:.4f}, epoch={}".format(
                vloss.mean().asscalar(), ploss.mean().asscalar(), epoch))
        if epoch%EPOCH_PER_BACKUP == 0:
            net.backup()
        loss.backward()
        trainer.step(len(Vs))

def play(p1, p2, max_turn=None, silent=False):
    colorMap = {Black: p1, White: p2, Empty: None}
    game = State()
    turn = 0
    while not (game.lastPlayerWin() or len(game.optionMoves) == 0):
        move = colorMap[game.nowPlayer].play(game)
        game.next(move[0], move[1])
        turn += 1
        if not silent:
            print("*** ROUND {:03d} ***\n{}\n".format(turn, game))
        if not(max_turn is None) and turn >= max_turn:
            return Empty, turn
    if game.lastPlayerWin():
        winner = game.nowPlayer.flip()
        if not silent:
            print("{}/{} win!".format(colorMap[winner].name(), winner))
    else:
        winner = Empty
        if not silent:
            print("even!")
    return colorMap[winner], turn

def benchmark(p1, p2, test_game_num=10):
    p1_win, p2_win, even = 0, 0, 0
    for i in range(test_game_num):
        if i%2 == 0:
            winPlayer, rounds = play(p1, p2, silent=True)
        else:
            winPlayer, rounds = play(p2, p1, silent=True)
        if winPlayer is p1:
            p1_win += 1
        elif winPlayer is p2:
            p2_win += 1
        else:
            even += 1
        print('\rSCORE=> TOTAL#{:<5d} {}#{:<5d} {}#{:<5d} EVEN#{:<5d}'.format(
              i+1, p1.name(), p1_win, p2.name(), p2_win, even), end='', flush=True)
        p1.reset()
        p2.reset()
    
#net60 = PVNet("net60", verno=60)
#net680 = PVNet("net680", verno=680)
#p1 = MCTSPlayer(net60)
#p2 = MCTSPlayer(net680)
#p3 = HumanPlayer("human")
#p4 = RandPlayer("random1")
#p5 = RandPlayer("random2")
#play(p1, p2)
#benchmark(p1, p2)

train(PVNet("FIVE", verno=0))