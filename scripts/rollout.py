from util.rlkit_utils import simulate_policy
from util.arguments import add_rollout_args, parser
import robosuite as suite
from robosuite.wrappers import GymWrapper
from robosuite.controllers import ALL_CONTROLLERS, load_controller_config
import numpy as np
import torch
import imageio
import os
import json

from signal import signal, SIGINT
from sys import exit

os.environ['KMP_DUPLICATE_LIB_OK'] = "True"

# Add and parse arguments
add_rollout_args()
args = parser.parse_args()

# Define callbacks
video_writer = None


def handler(signal_received, frame):
    # Handle any cleanup here
    print('SIGINT or CTRL-C detected. Closing video writer and exiting gracefully')
    video_writer.close()
    exit(0)


# Tell Python to run the handler() function when SIGINT is recieved
signal(SIGINT, handler)

if __name__ == "__main__":
    # Set random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Get path to saved model
    kwargs_fpath = os.path.join(args.load_dir, "variant.json")
    try:
        with open(kwargs_fpath) as f:
            kwargs = json.load(f)
    except FileNotFoundError:
        print("Error opening default controller filepath at: {}. "
              "Please check filepath and try again.".format(kwargs_fpath))

    # Grab / modify env args
    env_args = kwargs["eval_environment_kwargs"]
    if args.horizon is not None:
        env_args["horizon"] = args.horizon
    env_args["render_camera"] = args.camera
    env_args["hard_reset"] = True
    env_args["ignore_done"] = True

    # Specify camera name if we're recording a video
    if args.record_video:
        env_args["camera_names"] = args.camera
        env_args["camera_heights"] = 512
        env_args["camera_widths"] = 512

    # Setup video recorder if necesssary
    if args.record_video:
        # Grab name of this rollout combo
        video_name = "{}-{}-{}".format(
            env_args["env_name"], "".join(env_args["robots"]), env_args["controller"]).replace("_", "-")
        # Calculate appropriate fps
        fps = int(env_args["control_freq"])
        # Define video writer
        video_writer = imageio.get_writer("{}.mp4".format(video_name), fps=fps)

    # Pop the controller
    controller = env_args.pop("controller")
    if controller in ALL_CONTROLLERS:
        controller_config = load_controller_config(default_controller=controller)
    else:
        controller_config = load_controller_config(custom_fpath=controller)

    # Create env
    env_suite = suite.make(**env_args,
                           controller_configs=controller_config,
                           has_renderer=not args.record_video,
                           has_offscreen_renderer=args.record_video,
                           use_object_obs=True,
                           use_camera_obs=args.record_video,
                           reward_shaping=True
                           )
    
    # Make sure we only pass in the proprio and object obs (no images)
    keys = ["object-state"]
    for idx in range(len(env_suite.robots)):
        keys.append(f"robot{idx}_proprio-state")
    
    # Wrap environment so it's compatible with Gym API
    env = GymWrapper(env_suite, keys=keys)

    # Run rollout
    simulate_policy(
        env=env,
        model_path=os.path.join(args.load_dir, "params.pkl"),
        horizon=env_args["horizon"],
        render=not args.record_video,
        video_writer=video_writer,
        num_episodes=args.num_episodes,
        printout=True,
        use_gpu=args.gpu,
    )
