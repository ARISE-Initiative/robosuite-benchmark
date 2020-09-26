"""
Utility functions for parsing / processing command line arguments
"""

import argparse
from util.rlkit_utils import AGENTS


# Define mapping from string True / False to bool True / False
BOOL_MAP = {
    "true" : True,
    "false" : False
}

# Define parser
parser = argparse.ArgumentParser(description='RL args using agents / algs from rlkit and envs from robosuite')

# Add seed arg always
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')


def add_robosuite_args():
    """
    Adds robosuite args to command line arguments list
    """
    parser.add_argument(
        '--env',
        type=str,
        default='Lift',
        help='Robosuite env to run test on')
    parser.add_argument(
        '--robots',
        nargs="+",
        type=str,
        default='Panda',
        help='Robot(s) to run test with')
    parser.add_argument(
        '--eval_horizon',
        type=int,
        default=500,
        help='max num of timesteps for each eval simulation')
    parser.add_argument(
        '--expl_horizon',
        type=int,
        default=500,
        help='max num of timesteps for each eval simulation')
    parser.add_argument(
        '--policy_freq',
        type=int,
        default=20,
        help='Policy frequency for environment (Hz)')
    parser.add_argument(
        '--controller',
        type=str,
        default="OSC_POSE",
        help='controller to use for robot environment. Either name of controller for default config or filepath to custom'
             'controller config')
    parser.add_argument(
        '--reward_scale',
        type=float,
        default=1.0,
        help='max reward from single environment step'
    )
    parser.add_argument(
        '--hard_reset',
        action="store_true",
        help='If set, uses hard resets for this env'
    )

    # Environment-specific arguments
    parser.add_argument(
        '--env_config',
        type=str,
        default=None,
        choices=['single-arm-parallel', 'single-arm-opposed', 'bimanual'],
        help='Robosuite env configuration. Only necessary for bimanual environments')
    parser.add_argument(
        '--prehensile',
        type=str,
        default=None,
        choices=["True", "False", "true", "false"],
        help='Whether to use prehensile config. Only necessary for TwoArmHandoff env'
    )


def add_agent_args():
    """
    Adds args necessary to define a general agent and trainer in rlkit
    """
    parser.add_argument(
        '--agent',
        type=str,
        default="SAC",
        choices=AGENTS,
        help='Agent to use for training')
    parser.add_argument(
        '--qf_hidden_sizes',
        nargs="+",
        type=int,
        default=[256, 256],
        help='Hidden sizes for Q network ')
    parser.add_argument(
        '--policy_hidden_sizes',
        nargs="+",
        type=int,
        default=[256, 256],
        help='Hidden sizes for policy network ')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.99,
        help='Discount factor')
    parser.add_argument(
        '--policy_lr',
        type=float,
        default=3e-4,
        help='Learning rate for policy')
    parser.add_argument(
        '--qf_lr',
        type=float,
        default=3e-4,
        help='Quality function learning rate')

    # SAC-specific
    parser.add_argument(
        '--soft_target_tau',
        type=float,
        default=5e-3,
        help='Soft Target Tau value for Value function updates')
    parser.add_argument(
        '--target_update_period',
        type=int,
        default=1,
        help='Number of steps between target updates')
    parser.add_argument(
        '--no_auto_entropy_tuning',
        action='store_true',
        help='Whether to automatically tune entropy or not (default is ON)')

    # TD3-specific
    parser.add_argument(
        '--target_policy_noise',
        type=float,
        default=0.2,
        help='Target noise for policy')
    parser.add_argument(
        '--policy_and_target_update_period',
        type=int,
        default=2,
        help='Number of steps between policy and target updates')
    parser.add_argument(
        '--tau',
        type=float,
        default=0.005,
        help='Tau value for training')


def add_training_args():
    """
    Adds training parameters used during the experiment run
    """
    parser.add_argument(
        '--variant',
        type=str,
        default=None,
        help='If set, will use stored configuration from the specified filepath (should point to .json file)')
    parser.add_argument(
        '--n_epochs',
        type=int,
        default=2000,
        help='Number of epochs to run')
    parser.add_argument(
        '--trains_per_train_loop',
        type=int,
        default=1000,
        help='Number of training steps to take per training loop')
    parser.add_argument(
        '--expl_ep_per_train_loop',
        type=int,
        default=10,
        help='Number of exploration episodes to take per training loop')
    parser.add_argument(
        '--steps_before_training',
        type=int,
        default=1000,
        help='Number of exploration steps to take before starting training')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=256,
        help='Batch size per training step')
    parser.add_argument(
        '--num_eval',
        type=int,
        default=10,
        help='Num eval episodes to run for each trial run')
    parser.add_argument(
        '--log_dir',
        type=str,
        default='../log/runs/',
        help='directory to save runs')


def add_rollout_args():
    """
    Adds rollout arguments needed for evaluating / visualizing a trained rlkit policy
    """
    parser.add_argument(
        '--load_dir',
        type=str,
        required=True,
        help='path to the snapshot directory folder')
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=10,
        help='Num rollout episodes to run')
    parser.add_argument(
        '--horizon',
        type=int,
        default=None,
        help='Horizon to use for rollouts (overrides default if specified)')
    parser.add_argument(
        '--gpu',
        action='store_true',
        help='If true, uses GPU to process model')
    parser.add_argument(
        '--camera',
        type=str,
        default='frontview',
        help='Name of camera for visualization')
    parser.add_argument(
        '--record_video',
        action='store_true',
        help='If set, will save video of rollouts')


def get_expl_env_kwargs(args):
    """
    Grabs the robosuite-specific arguments and converts them into an rlkit-compatible dict for exploration env
    """
    env_kwargs = dict(
        env_name=args.env,
        robots=args.robots,
        horizon=args.expl_horizon,
        control_freq=args.policy_freq,
        controller=args.controller,
        reward_scale=args.reward_scale,
        hard_reset=args.hard_reset,
        ignore_done=True,
    )

    # Add in additional ones that may not always be specified
    if args.env_config is not None:
        env_kwargs["env_configuration"] = args.env_config
    if args.prehensile is not None:
        env_kwargs["prehensile"] = BOOL_MAP[args.prehensile.lower()]

    # Lastly, return the dict
    return env_kwargs


def get_eval_env_kwargs(args):
    """
    Grabs the robosuite-specific arguments and converts them into an rlkit-compatible dict for evaluation env
    """
    env_kwargs = dict(
        env_name=args.env,
        robots=args.robots,
        horizon=args.eval_horizon,
        control_freq=args.policy_freq,
        controller=args.controller,
        reward_scale=1.0,
        hard_reset=args.hard_reset,
        ignore_done=True,
    )

    # Add in additional ones that may not always be specified
    if args.env_config is not None:
        env_kwargs["env_configuration"] = args.env_config
    if args.prehensile is not None:
        env_kwargs["prehensile"] = BOOL_MAP[args.prehensile.lower()]

    # Lastly, return the dict
    return env_kwargs
