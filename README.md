# OpenAI_SpiningUP_for_dummy
Representative example of using OpenAI SpinningUP repo with custom gym environment

## Introduction to OpenAI-SpinningUP
&#160;&#160;&#160;&#160; **SpinningUP** is a deep reinforcement learning (RL) project proposed by OpenAI recently. In my opinion, both the API and source code is very elegant and much easier to use than Baseline project, especially for dummy like myself.

### Main framework of SpinningUP
&#160;&#160;&#160;&#160; **SpinningUP** seperates a typical RL task into two parts: environment(Env) and agent(algorithm). Env module is responsible for the interaction between agent and outer-environment and always follows the same API of OpenAI Gym. Agent module contains multiple great RL algorithm (DDPG, TRPO, PPO, etc) and is resposible for the training on given env. In the following, we show an example which deploys a agent (PPO) on a self-defined env.

#### Custom env
&#160;&#160;&#160;&#160; 
