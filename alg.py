from parl.core.fluid import layers
from copy import deepcopy
from paddle import fluid
from parl.core.fluid.algorithm import Algorithm

__all__ = ['GSMADDPG']

from parl.core.fluid.policy_distribution import SoftCategoricalDistribution
from parl.core.fluid.policy_distribution import SoftMultiCategoricalDistribution


def SoftPDistribution(logits, act_space):
    """Args:
            logits: the output of policy model
            act_space: action space, must be gym.spaces.Discrete or multiagent.multi_discrete.MultiDiscrete

        Return:
            instance of SoftCategoricalDistribution or SoftMultiCategoricalDistribution
    """
    # is instance of gym.spaces.Discrete
    if (hasattr(act_space, 'n')):
        return SoftCategoricalDistribution(logits)
    # is instance of multiagent.multi_discrete.MultiDiscrete
    elif (hasattr(act_space, 'num_discrete_space')):
        return SoftMultiCategoricalDistribution(logits, act_space.low,
                                                act_space.high)
    else:
        raise AssertionError("act_space must be instance of \
            gym.spaces.Discrete or multiagent.multi_discrete.MultiDiscrete")


class GSMADDPG(Algorithm):
    def __init__(self,
                 model,
                 agent_index=None,
                 act_space=None,
                 gamma=None,
                 tau=None,
                 lr=None):

        self.agent_index = agent_index
        self.act_space = act_space
        self.gamma = gamma
        self.tau = tau
        self.lr = lr

        self.model = model
        self.target_model = deepcopy(model)

    def predict(self, obs):
        """ input:
                obs: observation, shape([B] + shape of obs_n[agent_index])
            output:
                act: action, shape([B] + shape of act_n[agent_index])
        """
        this_policy = self.model.policy(obs)
        this_action = SoftPDistribution(
            logits=this_policy,
            act_space=self.act_space[self.agent_index]).sample()
        return this_action

    def predict_next(self, obs):
        """ input:  observation, shape([B] + shape of obs_n[agent_index])
            output: action, shape([B] + shape of act_n[agent_index])
        """
        next_policy = self.target_model.policy(obs)
        next_action = SoftPDistribution(
            logits=next_policy,
            act_space=self.act_space[self.agent_index]).sample()
        return next_action

    def Q(self, obs_n, act_n):
        """ input:
                obs_n: all agents' observation, shape([B] + shape of obs_n)
            output:
                act_n: all agents' action, shape([B] + shape of act_n)
        """
        return self.model.value(obs_n, act_n)

    def Q_next(self, obs_n, act_n):
        """ input:
                obs_n: all agents' observation, shape([B] + shape of obs_n)
            output:
                act_n: all agents' action, shape([B] + shape of act_n)
        """
        return self.target_model.value(obs_n, act_n)

    def learn(self, obs_n, act_n, target_q):
        """ update actor and critic model with MADDPG algorithm
        """
        actor_cost = self._actor_learn(obs_n, act_n)
        critic_cost = self._critic_learn(obs_n, act_n, target_q)
        return critic_cost

    def _actor_learn(self, obs_n, act_n):
        i = self.agent_index
        this_policy = self.model.policy(obs_n[i])
        sample_this_action = SoftPDistribution(
            logits=this_policy,
            act_space=self.act_space[self.agent_index]).sample()

        action_input_n = act_n + []
        action_input_n[i] = sample_this_action
        eval_q = self.Q(obs_n, action_input_n)
        act_cost = layers.reduce_mean(-1.0 * eval_q)

        act_reg = layers.reduce_mean(layers.square(this_policy))

        cost = act_cost + act_reg * 1e-3

        fluid.clip.set_gradient_clip(
            clip=fluid.clip.GradientClipByNorm(clip_norm=0.5),
            param_list=self.model.get_actor_params())

        optimizer = fluid.optimizer.AdamOptimizer(self.lr)
        optimizer.minimize(cost, parameter_list=self.model.get_actor_params())
        return cost

    def _critic_learn(self, obs_n, act_n, target_q):
        pred_q = self.Q(obs_n, act_n)
        cost = layers.reduce_mean(layers.square_error_cost(pred_q, target_q))

        fluid.clip.set_gradient_clip(
            clip=fluid.clip.GradientClipByNorm(clip_norm=0.5),
            param_list=self.model.get_critic_params())

        optimizer = fluid.optimizer.AdamOptimizer(self.lr)
        optimizer.minimize(cost, parameter_list=self.model.get_critic_params())
        return cost

    def sync_target(self, decay=None):
        if decay is None:
            decay = 1.0 - self.tau
        self.model.sync_weights_to(self.target_model, decay=decay)
