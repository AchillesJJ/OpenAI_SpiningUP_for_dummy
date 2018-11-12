# OpenAI_SpiningUP_for_dummy
Representative example of using OpenAI SpinningUP repo with custom gym environment

## Introduction to OpenAI-SpinningUP
&#160;&#160;&#160;&#160; **SpinningUP** is a deep reinforcement learning (RL) project proposed by OpenAI recently. In my opinion, both the API and source code is very elegant and much easier to use than Baseline project, especially for dummy like myself.

### Main framework of SpinningUP
&#160;&#160;&#160;&#160; **SpinningUP** seperates a typical RL task into two parts: environment(Env) and agent(algorithm). Env module is responsible for the interaction between agent and outer-environment and always follows the same API of OpenAI Gym. Agent module contains multiple great RL algorithm (DDPG, TRPO, PPO, etc) and is resposible for the training on given env. In the following, we show an example which deploys a agent (PPO) on a self-defined env.

#### Custom env
Custom env must have be registered by Gym and has the same API as Gym.
+ To be registered by Gym, youe custom env module should have at least the following files (structures):
```
gym-foo/
  README.md
  setup.py
  gym_foo/
    __init__.py
    envs/
      __init__.py
    foo_env.py
```
+ `gym-foo/setup.py` should have:
```
# used for env setup and dependencies install
from setuptools import setup

setup(
  name='gym_foo',
  version='0.0.1',
  install_requires=['gym'] # and any other dependencies that foo needs
)
```
+ `gym-foo/gym_foo/__init__.py` should have:
```
# register your env to Gym so we can use common API later
from gym.envs.registration import register

register(
  id='foo-v0', # name in gym.make('foo-v0')
  entry_point='gym_foo.envs:FooEnv',
)
```
+ `gym-foo/gym_foo/envs/__init__.py` should have:
```
# just relative import
from gym_foo envs.foo_env import FooEnv
```
+ Important part: custom env `gym-foo/gym_foo/envs/foo_env.py` must have the same API as Gym. Thus it should at least realize three major methods `reset`, `step` and `render`. A typical env should look like:
```
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class FooEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self):
    ...
  def step(self, action):
    ...
  def reset(self):
    ...
  def render(self, mode='human', close=False):
    ...
```
+ Once all is done, we can just `cd gym-foo/` and install the custom environment with pip `pip3 install -e .`. This will enable us to use the custom just as Gym itself. For example,
```
import gym
import gym_foo

env = gym.make('foo-v0')
ob = env.reset()
ob2, r, done, _ = env.step(action)
...
```

#### Launching agent from scripts
&#160;&#160;&#160;&#160; **SpinningUP** supports both command-line and scripts. Since we can see more details in script way, we will show a script example only in the following.
+ Script example: deploy a `ppo` agent on out custom env `foo-v0`
```
# encoding: utf-8
from spinup import ppo
import tensorflow as tf
import gym
import gym_foo
# choose `foo-v0` from Gym
env = gym.make('foo-v0')
env_fn = lambda: env

# define hyper-parameters for actor-critic neural network
ac_kwargs = dict(hidden_sizes=[64,64], activation=tf.nn.relu, output_activation=tf.nn.tanh)

# define parameter for logger
logger_kwargs = dict(output_dir='outputs', exp_name='trial_PPO')

# deploy ppo on foo-v0, run 200 epochs
ppo(env_fn=env_fn, ac_kwargs=ac_kwargs, steps_per_epoch=5000, epochs=200, logger_kwargs=logger_kwargs)
```
+ Once the training is over, the best model will be stored in given `output_dir` which always have the following files:
```

```
