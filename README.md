# robosuite v1.3 Benchmarking
Welcome to the robosuite v1.3 benchmarking repository! This repo is intended to provide a skeleton for SAC benchmarking experiments using our identical training environment.

NOTE: Because of dynamics changes occurring between robosuite v1.0 and v1.3, this branch (which uses v1.3) is not expected to produce identical results when deploying our pretrained policies (which were trained on v1.0). To
visualize our original benchmarked policy rollouts, please switch to the [v1.0 branch](https://github.com/ARISE-Initiative/robosuite-benchmark/tree/v1.0).

## Getting Started
Our benchmark consists of training [Soft Actor-Critic](https://arxiv.org/abs/1812.05905) agents implemented from [rlkit](https://github.com/vitchyr/rlkit). We built on top of rlkit's standard functionality to provide extra features useful for our purposes, such as video recording of rollouts and asymmetrical exploration / evaluation horizons.

To begin, start by cloning this repository from your terminal and moving into this directory:
```bash
$ git clone https://github.com/ARISE-Initiative/robosuite-benchmark.git
$ cd robosuite-benchmark
```

Our benchmarking environment consists of a Conda-based Python virtual environment running Python 3.8.8, and is supported for Mac OS X and Linux. Other versions / machine configurations have not been tested. [Conda](https://docs.conda.io/en/latest/) is a useful tool for creating virtual environments for Python, and can be installed [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/).

After installing Conda, create a new python virtual environment, activate this environment, and install this repo:

```bash
$ conda create -y -n rb_bench python=3.8
$ source activate rb_bench
$ (rb_bench) pip install -e .
```

Next, we must install pytorch. We will install using Conda so we can automatically account for CUDA dependencies. If you are running linux, run:
```bash
$ (rb_bench) conda install pytorch cudatoolkit=11.3 -c pytorch
```

If you are running Mac, instead run:
```bash
$ (rb_bench) conda install pytorch -c pytorch
```

Next, we must install rlkit. Go the the [rlkit](https://github.com/vitchyr/rlkit) repository and clone and install it, in your preferred directory. Note that we currently require a specific rlkit version as the current release is incompatible with our repo:
```bash
$ (rb_bench) cd <PATH_TO_YOUR_RLKIT_LOCATION>
$ (rb_bench) git clone https://github.com/rail-berkeley/rlkit.git
$ (rb_bench) cd rlkit
$ (rb_bench) git reset --hard b7f97b2463df1c5a1ecd2d293cfcc7a4971dd0ab
$ (rb_bench) pip install -e .
```

Lastly, if you are running Linux, you must make sure libglew is installed and add the libGLEW path to `LD_PRELOAD` environment variable (see [this thread](https://github.com/openai/mujoco-py/issues/268#issuecomment-402803943)):
```bash
(rb_bench) sudo apt-get update -y
(rb_bench) sudo apt-get install -y libglew-dev
(rb_bench) export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

## Running an Experiment
To validate our results on your own machine, or to experiment with another set of hyperparameters, we provide a [training script](rb_bench/scripts/train.py) as an easy entry point for executing individual experiments. For a given training run, a configuration must be specified -- this can be done in one of two ways:

1. **Command line arguments.** It may be useful to specify your desired configuration on the fly, from the command line. However, as there are many potential arguments that can be provided for training, we have modularized and organized them within a separate [arguments](rb_bench/util/arguments.py) module that describes all potential arguments for a given script. Note that for this training script, the `robosuite`, `agent`, and `training_args` are relevant here. Note that there are default values already specified for most of these values.

2. **Configuration files.** It is often more succinct and efficient to specify a configuration file (`.json`), and load this during runtime for training. If the `--variant` argument is specified, the configuration will be loaded and used for training. In this case, the resulting script execution line will look like so:

```bash
$ (rb_bench) python rb_bench/scripts/train.py --variant <PATH_TO_CONFIG>.json
```

This is also a useful method for automatically validating our benchmarking experiments on your own machine, as every experiment's configuration is saved and provided on this repo. For an example of the structure and values expected within a given configuration file, please see [this example](runs/Door-Panda-OSC-POSE-SEED17/Door_Panda_OSC_POSE_SEED17_2020_09_13_00_26_44_0000--s-0/variant.json).

Note that, by default, all training runs are stored in `log/runs/` directory, though this location may be changed by setting a different file location with the `--log_dir` flag.


## Visaulizing Training
During training, you can visualize current logged runs using viskit (see [Getting Started](#getting-started)). For visualizing active runs, we utilize rlkit's extraction of [rllab](https://github.com/rll/rllab)'s [viskit](https://github.com/vitchyr/viskit) package. We must download this repo and create a separate python environment (using Python 3.7) to install viskit::
```bash
$ conda create -y -n viskit python=3.7
$ conda activate viskit
$ (viskit) cd <PATH_TO_YOUR_VISKIT_LOCATION>
$ (viskit) git clone https://github.com/vitchyr/viskit.git
$ (viskit) cd viskit
$ (viskit) pip install -e .

```

Once viskit is installed and configured, you can easily see your results as follows at port 5000 in your browser:

```bash
$ (viskit) python <PATH_TO_VISKIT_DIR>/viskit/frontend.py <PATH_TO_LOG_DIR>
```

## Visualizing Rollouts
We provide a [rollout](rb_bench/scripts/rollout.py) script for executing and visualizing rollouts using a trained agent model. The relevant command-line arguments that can be specified for this script are the `rollout` args in the `util/arguments.py` module. Of note:

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
