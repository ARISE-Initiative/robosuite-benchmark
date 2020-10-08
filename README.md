# robosuite v1.0 Benchmarking
Welcome to the robosuite v1.0 benchmarking repository! This repo is intended for ease of replication of our benchmarking results, as well as providing a skeleton for further experiments or benchmarking using our identical training environment.

## Getting Started
Our benchmark consists of training [Soft Actor-Critic](https://arxiv.org/abs/1812.05905) agents implemented from [rlkit](https://github.com/vitchyr/rlkit). We built on top of rlkit's standard functionality to provide extra features useful for our purposes, such as video recording of rollouts and asymmetrical exploration / evaluation horizons.

To begin, start by cloning this repository from your terminal and moving into this directory:
```bash
$ git clone https://github.com/ARISE-Initiative/robosuite-benchmark.git
$ cd robosuite-benchmark
```

Our benchmarking environment consists of a Conda-based Python virtual environment running Python 3.7.4, and is supported for Mac OS X and Linux. Other versions / machine configurations have not been tested. [Conda](https://docs.conda.io/en/latest/) is a useful tool for creating virtual environments for Python, and can be installed [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

After installing Conda, create a new virtual environment using our pre-configured environment setup, and activate this environment. Note that we have to unfortunately do a two-step installation process in order to avoid some issues with precise versions:

```bash
$ conda env create -f environments/rb_bench_[linux/mac]_env.yml
$ source activate rb_bench
$ pip install -r requirements.txt
```

Next, we must install rlkit. Go the the [rlkit](https://github.com/vitchyr/rlkit) repository and clone and install it, in your preferred directory. Note that we currently require a specific rlkit version as the current release is incompatible with our repo:
```bash
$ (rb_bench) cd <PATH_TO_YOUR_RLKIT_LOCATION>
$ (rb_bench) git clone https://github.com/vitchyr/rlkit.git
$ (rb_bench) cd rlkit
$ (rb_bench) git reset --hard f136e140a57078c4f0f665051df74dffb1351f33
$ (rb_bench) pip install -e .
```

Lastly, for visualizing active runs, we utilize rlkit's extraction of [rllab](https://github.com/rll/rllab)'s [viskit](https://github.com/vitchyr/viskit) package:
```bash
$ (rb_bench) cd <PATH_TO_YOUR_VISKIT_LOCATION>
$ (rb_bench) git clone https://github.com/vitchyr/viskit.git
$ (rb_bench) cd viskit
$ (rb_bench) pip install -e .

```

## Running an Experiment
To validate our results on your own machine, or to experiment with another set of hyperparameters, we provide a [training script](scripts/train.py) as an easy entry point for executing individual experiments. Note that this repository must be added to your `PYTHONPATH` before running any scripts; this can be done like so:

```bash
$ (rb_bench) cd <PATH_TO_YOUR_ROBOSUITE_BENCHMARKING_REPO_DIR>
$ (rb_bench) export PYTHONPATH=.:$PYTHONPATH
```

For a given training run, a configuration must be specified -- this can be done in one of two ways:

1. **Command line arguments.** It may be useful to specify your desired configuration on the fly, from the command line. However, as there are many potential arguments that can be provided for training, we have modularized and organized them within a separate [arguments](util/arguments.py) module that describes all potential arguments for a given script. Note that for this training script, the `robosuite`, `agent`, and `training_args` are relevant here. Note that there are default values already specified for most of these values.

2. **Configuration files.** It is often more succinct and efficient to specify a configuration file (`.json`), and load this during runtime for training. If the `--variant` argument is specified, the configuration will be loaded and used for training. In this case, the resulting script execution line will look like so:

```bash
$ (rb_bench) python scripts/train.py --variant <PATH_TO_CONFIG>.json
```

This is also a useful method for automatically validating our benchmarking experiments on your own machine, as every experiment's configuration is saved and provided on this repo. For an example of the structure and values expected within a given configuration file, please see [this example](runs/Door-Panda-OSC-POSE-SEED17/Door_Panda_OSC_POSE_SEED17_2020_09_13_00_26_44_0000--s-0/variant.json).

Note that, by default, all training runs are stored in `log/runs/` directory, though this location may be changed by setting a different file location with the `--log_dir` flag.


## Visaulizing Training
During training, you can visualize current logged runs using viskit (see [Getting Started](#getting-started)). Once viskit is installed and configured, you can easily see your results as follows at port 5000 in your browser:

```bash
$ (rb_bench) python <PATH_TO_VISKIT_DIR>/viskit/frontend.py <PATH_TO_LOG_DIR>
```

## Visualizing Rollouts
We provide a [rollout](scripts/rollout.py) script for executing and visualizing rollouts using a trained agent model. The relevant command-line arguments that can be specified for this script are the `rollout` args in the `util/arguments.py` module. Of note:

* `load_dir` specifies the path to the logging directory that contains both the `variant.json` and `params.pkl` specifying the training configuration and agent model, respectively,

* `camera` specifies the robosuite-specific camera to use for rendering images / video (`frontview` and `agentview` are common choices),

* `record_video` specifies whether to save a video of the resulting rollouts (note that if this is set, no onscreen renderer will be used!)

A simple example for using this rollout script can be seen as follows:

```bash
$ (rb_bench) python scripts/rollout.py --load_dir runs/Door-Panda-OSC-POSE-SEED17/Door_Panda_OSC_POSE_SEED17_2020_09_13_00_26_44_0000--s-0/ --horizon 200 --camera frontview
```

This will execute the trained model configuration used in our benchmarking from the `Door` environment with `Panda` / `OSC_POSE` using seed 17, without rollouts only occurring up to 200 timesteps per episode and using the `frontview` camera for visualization.

## Problems?
For any problems encountered when running this repo, please submit an issue!
