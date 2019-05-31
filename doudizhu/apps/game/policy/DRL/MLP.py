import mxnet as mx
from mxnet import gluon, init, autograd, nd
from mxnet.gluon import loss as gloss
from mxnet.gluon import nn
import sys

# TODO change this!!!

class MultiLevelPerceptron(object):

    def __init__(self, 
        action_dim, 
        hidden_dims=(200,200,200,100), 
        learning_rate=1e-3,
        freeze=False):

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
        self.ctx = try_gpu()
        self.net = nn.Sequential()
        for d in hidden_dims:
            self.net.add(
                nn.Dense(d, activation='relu'),
                nn.BatchNorm(),
            )
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
            print("From '{}' Load DQN failed - {}".format(loc, str(err)))

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
        return "MLP-lr({})-{}-e({})".format(
            self.learning_rate,
            '-'.join(['h{}({})'.format(i+1, d) for i, d in enumerate(self.hidden_dims)]),
            self.train_num
        )
