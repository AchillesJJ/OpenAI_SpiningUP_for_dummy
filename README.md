# OpenAI_SpiningUP_for_dummy
Representative example of using OpenAI SpinningUP repo with custom gym environment

## Introduction to OpenAI-SpinningUP
&#160;&#160;&#160;&#160; **SpinningUP** is a deep reinforcement learning (RL) project proposed by OpenAI recently. In my opinion, both the API and source code is very elegant and much easier to use than Baseline project, especially for dummy like myself.

### Main framework of SpinningUP
&#160;&#160;&#160;&#160; **SpinningUP** seperates a typical RL task into two parts: environment(Env) and agent(algorithm). Env module is responsible for the interaction between agent and outer-environment and always follows the same API of OpenAI Gym. Agent module contains multiple great RL algorithm (DDPG, TRPO, PPO, etc) and is resposible for the training on given env. In the following, we show an example which deploys a agent (PPO) on a self-defined env.

#### Custom env
Custom env must have be registered by Gym and has the same API as Gym.
+ To be registered by Gym, your custom env module should have at least the following files (structures):
```python
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
```python
# used for env setup and dependencies install
from setuptools import setup

setup(
    name='gym_foo',
    version='0.0.1',
    install_requires=['gym'] # and any other dependencies that foo needs
)
```
+ `gym-foo/gym_foo/__init__.py` should have:
```python
# register your env to Gym so we can use common API later
from gym.envs.registration import register

register(
    id='foo-v0', # name in gym.make('foo-v0')
    entry_point='gym_foo.envs:FooEnv',
)
```
+ `gym-foo/gym_foo/envs/__init__.py` should have:
```python
# just relative import
from gym_foo.envs.foo_env import FooEnv
```
+ Important part: custom env `gym-foo/gym_foo/envs/foo_env.py` must have the same API as Gym. Thus it should at least realize three major methods `reset`, `step` and `render`. A typical env should look like:
```python
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
```python
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
```python
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
```python
outputs/
  config.json
  process.txt
  simple_save/
    model_info.pkl
    saved_model.pb
    variables/
      variables.data-00000-of-00001
      variables.index
```
&#160;&#160;&#160;&#160; `config.json` and `process.txt` contains the model configuration and outputs during training process. Though **SpinningUP** provides easy ways, such as `test_policy` to check the policy model, I still recommand to restore the model by `tensorflow` from scratch. Everything needed to restore the trained agent is contained in the file `simple_save`, and we can restore it by using `tf.saved_model.loader.load`. Before the loading, we shall `saved_model_cli` command-line provided by `tensorflow`
```shell
saved_model_cli show --dir simple_save --all
```
&#160;&#160;&#160;&#160; then we got
```shell
MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['x'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 2)
        name: Placeholder:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['pi'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 1)
        name: pi/add:0
    outputs['v'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1)
        name: v/Squeeze:0
  Method name is: tensorflow/serving/predict
```
&#160;&#160;&#160;&#160; From the information given above, we can know the details of model. For example, the `tag` of the model is `serve`, the input node name is `Placeholder:0` and output node of policy network is `pi/add:0`. All of these are important for us to rebuild and run the saved model. Once we have these, we can now do the following
```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

export_dir = './simple_save' # model path

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ['serve'], export_dir) # load model
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name('Placeholder:0') # get the input node tensor by name
    q = graph.get_tensor_by_name('pi/add:0') # get the output node tensor of policy by name
    # policy outputs
    ls = []
    dim = 80
    for i in range(80):
        for j in range(80):
            rho_0 = (1.0/80)*(i+1)
            theta = (1.0/80)*(i+1)
            ls.append([rho_0, theta])
    pi = sess.run(q, feed_dict={x:ls}) # feed and evaluate the network

pi = np.array(pi)
pi = pi[:, 0]
pi = np.reshape(pi, (80, 80)).transpose()
plt.matshow(pi)
plt.show()
```

```python
class Alpha(AlphaBase):

    def init(self):
        ...
    def run_daily(self, di):
        ...
    def _register(self):
        ...
    def _synchronize(self, di):
        ...
    def _roll_train(self, epochs):
        ...
```

