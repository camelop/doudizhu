import mxnet as mx
from mxnet import gluon, init, autograd, nd
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn
import sys

# TODO change this!!!

class MultiLevelPerceptron(object):

    def __init__(self, action_dim, 
        h1_dim=1000,
        h2_dim=1000,
        learning_rate=1e-3):

        def try_gpu():
            try:
                ctx = mx.gpu()
                _ = nd.zeros((1,), ctx=ctx)
                print("init DQN using GPU")
            except mx.base.MXNetError:
                ctx = mx.cpu()
            return ctx

        self.learning_rate = learning_rate
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.train_num = 0
        self.ctx = try_gpu()
        self.net = nn.Sequential()
        self.net.add(
            nn.Dense(h1_dim, activation='relu'),
            nn.BatchNorm(),
            nn.Dense(h2_dim, activation='relu'),
            nn.BatchNorm(),
            nn.Dense(action_dim)
        )
        self.net.initialize(init.Normal(sigma=0.01), ctx=self.ctx)
        self.loss = gloss.L2Loss()
        self.trainer = gluon.Trainer(self.net.collect_params(), 'sgd', {'learning_rate': learning_rate})
    
    def train(self, states, actions, sampled_Q):
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

    def predict(self, states):
        # deprecated
        states = nd.array(states, ctx=self.ctx)
        return self.net(states).asnumpy()

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

    def save(self, tag):
        self._save("models/"+tag+".mxnet")

    def load(self, tag):
        self._save("models/"+tag+".mxnet")

    def _load(self, loc):
        try:
            self.net.load_parameters(loc)
            # print("From '{}' Load DQN success :D".format(loc))
        except:
            pass
            print("From '{}' Load DQN failed".format(loc))

    def _save(self, loc):
        try:
            self.net.save_parameters(loc)
            # print("To '{}' Save DQN success :D".format(loc))
        except:
            pass
            print("To '{}' Save DQN failed".format(loc))
    
    def reset(self):
        self.net.initialize(init.Normal(sigma=0.01), force_reinit=True)
    
    def __str__(self):
        return "MLP-lr({})-h1({})-h2({})-e({})".format(
            self.learning_rate,
            self.h1_dim,
            self.h2_dim,
            self.train_num
        )

# model_test
if __name__ == '__main__':
    import numpy as np
    import random
    model = MultiLevelPerceptron(3, h1_dim=400, h2_dim=400)
    states = np.random.random((5, 600))
    # test forward
    print("state -> " + str(states))
    print("predict -> " + str(model.predict(states)))
    # test train
    output = np.random.random((5, 3))
    first = model.predict(states)
    epoches = 1000
    for i in range(epoches):
        actions = [random.randint(0, 2) for _ in range(5)]
        model.train(states, actions, output[range(5), actions])
    print("output -> ")
    print(output)
    print("init ->")
    print(first )
    print("after {} epoches ->".format(epoches))
    print(model.predict(states))
    print("after reset ->".format(epoches))
    model.reset()
    print(model.predict(states))
