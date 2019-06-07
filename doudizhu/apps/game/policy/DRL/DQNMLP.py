import mxnet as mx
from mxnet import gluon, init, autograd, nd
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn
import sys

from .MLP import MultiLevelPerceptron

class DQNMLP(MultiLevelPerceptron):

    def __init__(self, 
        action_dim, 
        hidden_dims=(200,200,200,100), 
        learning_rate=1e-3,
        freeze=False,
        activation="relu"):
        super.__init__(self, action_dim, hidden_dims, learning_rate, freeze, activation)
        self.net.add(
            nn.Dense(action_dim)
        )
        self.net.initialize(init.Normal(sigma=0.01), ctx=self.ctx)
        self.loss = gloss.L2Loss()
        self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': learning_rate})
    
    def train(self, states, actions, sampled_Q):
        if not self.freeze:
            self.train_num += 1
            batch_size = len(states)
            states = nd.array(states, ctx=self.ctx)
            actions = nd.array(actions, ctx=self.ctx)
            sampled_Q = nd.array(sampled_Q, ctx=self.ctx)
            with autograd.record():
                Q = self.net(states)
                l = self.loss(Q[list(range(sampled_Q.shape[0])), actions], sampled_Q).sum()
            l.backward()
            # print 'loss -> {}'.format(l)
            self.trainer.step(batch_size)
            return l.asscalar()

    def value(self, vec, mask):
        # np.max(@all_allowed_Q)
        vec = nd.array(vec, ctx=self.ctx).reshape((1, -1))
        mask = nd.array(mask, ctx=self.ctx).reshape((1, -1))
        result = self.net(vec)[0]
        return nd.max(result[mask[0] > 0]).asscalar()
    
    def choose(self, vec, mask):
        vec = nd.array(vec, ctx=self.ctx).reshape((1, -1))
        mask = nd.array(mask, ctx=self.ctx).reshape((1, -1))
        result = (self.net(vec) + (mask - 1) * sys.maxsize)
        ret = int(nd.argmax(result, axis=1).asscalar())
        return ret