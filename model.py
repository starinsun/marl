import paddle.fluid as fluid
import parl
from parl import layers


class MAModel(parl.Model): # 从parl.Model继承model类
    def __init__(self, act_dim):
        self.actor_model = ActorModel(act_dim)   # model.actor_model即为策略网络，输入是action的维度
        self.critic_model = CriticModel()        # model.critic_model即为价值网络，输出Q-function指导策略网络优化

    def policy(self, obs):
        return self.actor_model.policy(obs)      #

    def value(self, obs, act):
        return self.critic_model.value(obs, act)

    def get_actor_params(self):
        return self.actor_model.parameters()

    def get_critic_params(self):
        return self.critic_model.parameters()


class ActorModel(parl.Model):                    #
    def __init__(self, act_dim):
        # 定义了三层网络的架构
        self.fc1 = layers.fc(
            size=64,
            act='relu',
            param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
        self.fc2 = layers.fc(
            size=64,
            act='relu',
            param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
        self.fc3 = layers.fc(
            size=act_dim,
            act=None,
            param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))

    def policy(self, obs):
        hid1 = self.fc1(obs)
        hid2 = self.fc2(hid1)
        means = self.fc3(hid2)
        means = means
        return means


class CriticModel(parl.Model):
    def __init__(self):

        self.fc1 = layers.fc(
            size=64,
            act='relu',
            param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
        self.fc2 = layers.fc(
            size=64,
            act='relu',
            param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))
        self.fc3 = layers.fc(
            size=1,
            act=None,
            param_attr=fluid.initializer.Normal(loc=0.0, scale=0.1))

    def value(self, obs_n, act_n):
        inputs = layers.concat(obs_n + act_n, axis=1)
        hid1 = self.fc1(inputs)
        hid2 = self.fc2(hid1)
        Q = self.fc3(hid2)
        Q = layers.squeeze(Q, axes=[1])
        return Q