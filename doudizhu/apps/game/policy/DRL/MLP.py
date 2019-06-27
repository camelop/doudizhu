import mxnet as mx
from mxnet import gluon, init, autograd, nd
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn
import sys

class MultiLevelPerceptron(object):

    def __init__(self, 
        action_dim, 
        hidden_dims=(200,200,200,100), 
        learning_rate=1e-3,
        freeze=False,
        activation="relu",
        batchnorm=True):

        def try_gpu():
            try:
                ctx = mx.gpu()
                _ = nd.zeros((1,), ctx=ctx)
                print("init DQN using GPU")
            except mx.base.MXNetError:
                ctx = mx.cpu()
            return ctx
            
        self.freeze = freeze
        self.learning_rate = learning_rate
        self.hidden_dims = hidden_dims
        self.train_num = 0
        self.activation = "relu"
        self.ctx = try_gpu()
        self.net = nn.Sequential()
        for d in hidden_dims:
            self.net.add(
                nn.Dense(d, activation=self.activation),
            )
            if batchnorm:
                self.net.add(
                    nn.BatchNorm(),
                )
        # define output net, initialize, loss and trainer
    
    def train(self, states, actions, sampled_Q):
        raise NotImplementedError

    # ------------------ support functions ------------------
    
    def save(self, tag):
        self._save("models/"+tag+".mxnet")

    def _save(self, loc):
        try:
            self.net.save_parameters(loc)
            # print("To '{}' Save DQN success :D".format(loc))
        except:
            pass
            print("To '{}' Save MLP failed".format(loc))

    def load(self, tag):
        self._load("models/"+tag+".mxnet")

    def _load(self, loc):
        try:
            self.net.load_parameters(loc)
            if 'e(' in loc:
                p = loc.find('e(') + 2
                num = ''
                while loc[p] != ')':
                    num = num + loc[p]
                    p += 1
                self.train_num = int(num)
            # print("From '{}' Load DQN success :D".format(loc))
        except Exception as err:
            pass
            print("From '{}' Load MLP failed - {}".format(loc, str(err)))

    def reset(self):
        self.net.initialize(init.Normal(sigma=0.01), force_reinit=True)
    
    def __str__(self):
        return "MLP-lr({})-{}-{}-e({})".format(
            self.learning_rate,
            self.activation,
            '-'.join(['h{}({})'.format(i+1, d) for i, d in enumerate(self.hidden_dims)]),
            self.train_num
        )
