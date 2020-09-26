import os
import numpy as np

import copy
import json
import torch
import rlkit.torch.pytorch_util as ptu
from rlkit.launchers.launcher_util import setup_logger

from util.rlkit_utils import experiment
from util.arguments import *

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Add necessary command line args
add_robosuite_args()
add_agent_args()
add_training_args()

# Global vars
THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# Objective function
def run_experiment():
    # Define agent-specific arguments
    trainer_kwargs = None
    if args.agent == "SAC":
        trainer_kwargs = dict(
            discount=args.gamma,
            soft_target_tau=args.soft_target_tau,
            target_update_period=args.target_update_period,
            policy_lr=args.policy_lr,
            qf_lr=args.qf_lr,
            reward_scale=args.reward_scale,
            use_automatic_entropy_tuning=(not args.no_auto_entropy_tuning),
        )
    elif args.agent == "TD3":
        trainer_kwargs = dict(
            target_policy_noise=args.target_policy_noise,
            discount=0.99,
            reward_scale=args.reward_scale,
            policy_learning_rate=args.policy_lr,
            qf_learning_rate=args.qf_lr,
            policy_and_target_update_period=args.policy_and_target_update_period,
            tau=args.tau,
        )
    else:
        pass

    # Construct variant to train
    if args.variant is None:
        variant = dict(
            algorithm=args.agent,
            seed=args.seed,
            version="normal",
            replay_buffer_size=int(1E6),
            qf_kwargs=dict(
                hidden_sizes=args.qf_hidden_sizes,
            ),
            policy_kwargs=dict(
                hidden_sizes=args.policy_hidden_sizes,
            ),
            algorithm_kwargs=dict(
                num_epochs=args.n_epochs,
                num_eval_steps_per_epoch=args.eval_horizon * args.num_eval,
                num_trains_per_train_loop=args.trains_per_train_loop,
                num_expl_steps_per_train_loop=args.expl_horizon * args.expl_ep_per_train_loop,
                min_num_steps_before_training=args.steps_before_training,
                expl_max_path_length=args.expl_horizon,
                eval_max_path_length=args.eval_horizon,
                batch_size=args.batch_size,
            ),
            trainer_kwargs=trainer_kwargs,
            expl_environment_kwargs=get_expl_env_kwargs(args),
            eval_environment_kwargs=get_eval_env_kwargs(args),
        )
        # Set logging
        tmp_file_prefix = "{}_{}_{}_SEED{}".format(args.env, "".join(args.robots), args.controller, args.seed)
    else:
        # This is a variant we want to load
        # Attempt to load the json file
        try:
            with open(args.variant) as f:
                variant = json.load(f)
        except FileNotFoundError:
            print("Error opening specified variant json at: {}. "
                  "Please check filepath and try again.".format(variant))

        # Set logging
        tmp_file_prefix = "{}_{}_{}_SEED{}".format(variant["expl_environment_kwargs"]["env_name"],
                                                   "".join(variant["expl_environment_kwargs"]["robots"]),
                                                   variant["expl_environment_kwargs"]["controller"],
                                                   args.seed)
        # Set agent
        args.agent = variant["algorithm"]

    # Setup logger
    abs_root_dir = os.path.join(THIS_DIR, args.log_dir)
    tmp_dir = setup_logger(tmp_file_prefix, variant=variant, base_log_dir=abs_root_dir)
    ptu.set_gpu_mode(torch.cuda.is_available())  # optionally set the GPU (default=False

    # Run experiment
    experiment(variant, agent=args.agent)


if __name__ == '__main__':
    # First, parse args
    args = parser.parse_args()

    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Notify user we're starting run
    print('\n\n')
    print('------------- Running {} --------------'.format(args.agent))

    print('  Params: ')
    if args.variant is None:
        for key, value in args.__dict__.items():
            if key.startswith('__') or key.startswith('_'):
                continue
            print('    {}: {}'.format(key, value))
    else:
        print('    variant: {}'.format(args.variant))

    print('\n\n')

    # Execute run
    run_experiment()

    print('Finished run!')

