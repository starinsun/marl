import os
import time
import argparse
import numpy as np
from model import MAModel
from agent import MAAgent
from alg import GSMADDPG
from parl.env.multiagent_simple_env import MAenv
from parl.utils import logger, summary


def run_episode(env, agents):
    obs_n = env.reset()
    total_reward = 0
    agents_reward = [0 for _ in range(env.n)]
    steps = 0
    while True:
        steps += 1
        action_n = [agent.predict(obs) for agent, obs in zip(agents, obs_n)]
        next_obs_n, reward_n, done_n, _ = env.step(action_n)
        done = all(done_n)
        terminal = (steps >= args.max_step_per_episode)

        for i, agent in enumerate(agents):
            agent.add_experience(obs_n[i], action_n[i], reward_n[i],
                                 next_obs_n[i], done_n[i])

        obs_n = next_obs_n
        for i, reward in enumerate(reward_n):
            total_reward += reward
            agents_reward[i] += reward

        if done or terminal:
            break

        if args.show:
            time.sleep(0.1)
            env.render()

        if args.restore and args.show:
            continue

        for i, agent in enumerate(agents):
            critic_loss = agent.learn(agents)
            summary.add_scalar('critic_loss_%d' % i, critic_loss,
                               agent.global_train_step)

    return total_reward, agents_reward, steps


def train_agent():
    env = MAenv(args.env)
    logger.info('智能体数量: {}'.format(env.n))
    logger.info('状态空间: {}'.format(env.observation_space))
    logger.info('动作空间: {}'.format(env.action_space))
    logger.info('obs_shape_n: {}'.format(env.obs_shape_n))
    logger.info('act_shape_n: {}'.format(env.act_shape_n))
    for i in range(env.n):
        logger.info('agent {} obs_low:{} obs_high:{}'.format(
            i, env.observation_space[i].low, env.observation_space[i].high))
        logger.info('agent {} act_n:{}'.format(i, env.act_shape_n[i]))
        if ('low' in dir(env.action_space[i])):
            logger.info('agent {} act_low:{} act_high:{} act_shape:{}'.format(
                i, env.action_space[i].low, env.action_space[i].high,
                env.action_space[i].shape))
            logger.info('num_discrete_space:{}'.format(
                env.action_space[i].num_discrete_space))

    from gym import spaces
    from multiagent.multi_discrete import MultiDiscrete
    for space in env.action_space:
        assert (isinstance(space, spaces.Discrete)
                or isinstance(space, MultiDiscrete))

    agents = []
    for i in range(env.n):
        model = MAModel(env.act_shape_n[i])
        algorithm = GSMADDPG(
            model,
            agent_index=i,
            act_space=env.action_space,
            gamma=args.gamma,
            tau=args.tau,
            lr=args.lr)
        agent = MAAgent(
            algorithm,
            agent_index=i,
            obs_dim_n=env.obs_shape_n,
            act_dim_n=env.act_shape_n,
            batch_size=args.batch_size,
            speedup=(not args.restore))
        agents.append(agent)
    total_steps = 0
    total_episodes = 0

    episode_rewards = []
    agent_rewards = [[] for _ in range(env.n)]
    final_ep_rewards = []
    final_ep_ag_rewards = []

    if args.restore:
        for i in range(len(agents)):
            model_file = args.model_dir + '/agent_' + str(i) + '.ckpt'
            if not os.path.exists(model_file):
                logger.info('model file {} does not exits'.format(model_file))
                raise Exception
            agents[i].restore(model_file)

    t_start = time.time()
    logger.info('开始训练')
    while total_episodes <= args.max_episodes:
        ep_reward, ep_agent_rewards, steps = run_episode(env, agents)
        if args.show:
            print('episode {}, reward {}, steps {}'.format(
                total_episodes, ep_reward, steps))

        total_steps += steps
        total_episodes += 1
        episode_rewards.append(ep_reward)
        for i in range(env.n):
            agent_rewards[i].append(ep_agent_rewards[i])

        if total_episodes % args.stat_rate == 0:
            mean_episode_reward = np.mean(episode_rewards[-args.stat_rate:])
            final_ep_rewards.append(mean_episode_reward)
            for rew in agent_rewards:
                final_ep_ag_rewards.append(np.mean(rew[-args.stat_rate:]))
            use_time = round(time.time() - t_start, 3)
            logger.info(
                '步数: {}, 轮次: {}, 收益: {}, 时间: {}'.
                format(total_steps, total_episodes, mean_episode_reward, use_time))
            t_start = time.time()
            summary.add_scalar('mean_episode_reward/episode', mean_episode_reward, total_episodes)
            summary.add_scalar('mean_episode_reward/steps', mean_episode_reward, total_steps)
            summary.add_scalar('use_time/1000episode', use_time, total_episodes)

            # 存储模型
            if not args.restore:
                os.makedirs(os.path.dirname(args.model_dir), exist_ok=True)
                for i in range(len(agents)):
                    model_name = '/agent_' + str(i) + '.ckpt'
                    agents[i].save(args.model_dir + model_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--env',
        type=str,
        default='simple_tag')
    parser.add_argument(
        '--max_step_per_episode',
        type=int,
        default=50,
        help='maximum step per episode')
    parser.add_argument(
        '--max_episodes',
        type=int,
        default=25000)
    parser.add_argument(
        '--stat_rate',
        type=int,
        default=1000,
        help='statistical interval of save model or count reward')
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-4,
        help='learning rate for Adam optimizer')
    parser.add_argument(
        '--gamma', type=float, default=0.95, help='discount factor')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1024,
        help='number of episodes to optimize at the same time')
    parser.add_argument('--tau', type=int, default=0.01, help='soft update')
    parser.add_argument(
        '--show', action='store_true', default=True)
    parser.add_argument(
        '--restore',
        action='store_true',
        default=True)
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./model')

    args = parser.parse_args()
    train_agent()