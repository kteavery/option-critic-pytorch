#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

from deep_rl import *
import argparse


# Option-Critic
def option_critic_feature(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.num_workers = 5
    config.task_fn = lambda: Task(config.game, num_envs=config.num_workers)
    config.eval_env = Task(config.game)
    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, 0.001)
    config.network_fn = lambda: OptionCriticNet(FCBody(config.state_dim), config.action_dim, num_options=2)
    config.random_option_prob = LinearSchedule(1.0, 0.1, 1e4)
    config.discount = 0.99
    config.target_network_update_freq = 200
    config.rollout_length = 5
    config.termination_regularizer = 0.01
    config.entropy_weight = 0.01
    config.gradient_clip = 5
    run_steps(OptionCriticAgent(config))


def option_critic_pixel(lr=0.001, num_options=8, toybox=True, **kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config = Config()
    config.merge(kwargs)

    config.toybox = toybox
    config.num_workers = 16

    config.task_fn = lambda: Task(config.game, toybox=toybox, num_envs=config.num_workers)
    config.eval_env = Task(config.game, toybox=toybox)

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, lr=lr, alpha=0.99, eps=1e-5)
    config.network_fn = lambda: OptionCriticNet(NatureConvBody(), config.action_dim, num_options=num_options)
    config.random_option_prob = LinearSchedule(0.1)
    config.state_normalizer = ImageNormalizer()
    config.reward_normalizer = SignNormalizer()
    config.discount = 0.99
    config.target_network_update_freq = 10000
    config.rollout_length = 5
    config.gradient_clip = 5
    config.max_steps = int(2e7)
    config.entropy_weight = 0.01
    config.termination_regularizer = 0.01
    run_steps(OptionCriticAgent(config))


if __name__ == '__main__':
    mkdir('log')
    mkdir('tf_log')
    set_one_thread()
    random_seed()
    # -1 is CPU, a positive integer is the index of GPU
    Config.DEVICE = torch.device('cuda')
    game = 'AmidarNoFrameskip-v4'

    parser = argparse.ArgumentParser(description="Option Critic PyTorch")
    parser.add_argument('--lr',type=float, default=0.001, help='Learning rate') # original default=0.0001
    parser.add_argument('--num_options',type=int, default=8, help='Number of options') # original default=4
    parser.add_argument('--toybox',type=bool, default=True, help='Toybox or regular ALE env') 
    args = parser.parse_args()

    option_critic_pixel(lr=args.lr, num_options=args.num_options, toybox=args.toybox, game=game)