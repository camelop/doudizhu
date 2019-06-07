import mxnet as mx
from mxnet import gluon, init, autograd, nd
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn
import sys

from .MLP import MultiLevelPerceptron

class REINFORCE_MLP(MultiLevelPerceptron):

    def __init__(self, 
        action_dim, 
        hidden_dims=(200,200,200,100), 
        learning_rate=1e-3,
        freeze=False,
        activation="relu"):
        super().__init__(action_dim, hidden_dims, learning_rate, freeze, activation)
        self.net.add(
            nn.Dense(action_dim)
        )
        self.net.initialize(init.Normal(sigma=0.01), ctx=self.ctx)
        self.loss = gloss.SoftmaxCrossEntropyLoss()
        self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': learning_rate})
    
    def train(self, states, actions, vts):
        if not self.freeze:
            self.train_num += 1
            batch_size = len(states)
            states = nd.array(states, ctx=self.ctx)
            actions = nd.array(actions, ctx=self.ctx)
            vts = nd.array(vts, ctx=self.ctx)
            print(states.shape)
            print(actions.shape)
            print(vts.shape)
            with autograd.record():
                pred = self.net(states)
                print(pred.shape)
                l = self.loss(pred, actions, vts).sum() # softmax_crossentropy
            l.backward()
            # print 'loss -> {}'.format(l)
            self.trainer.step(batch_size)
            return l.asscalar()
    
    def choose(self, vec, mask, rand):
        omask = mask
        vec = nd.array(vec, ctx=self.ctx).reshape((1, -1))
        mask = nd.array(mask, ctx=self.ctx).reshape((1, -1))
        result = list(nd.softmax(self.net(vec) + (mask - 1) * sys.maxsize, axis=1).asnumpy()[0])
        s = 0.0
        for i, cur in enumerate(result):
            s += cur
            if s > rand:
                assert omask[i] == 1
                return i
        return len(omask) - 1

    def __str__(self):
        return "REINFORCE-lr({})-{}-{}-e({})".format(
            self.learning_rate,
            self.activation,
            '-'.join(['h{}({})'.format(i+1, d) for i, d in enumerate(self.hidden_dims)]),
            self.train_num
        )
