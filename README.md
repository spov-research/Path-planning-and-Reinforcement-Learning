This is a fork of the [original repository](https://github.com/DavidCastilloAlvarado/Path-planning-and-Reinforcement-Learning.git) under the project [Smart_POV](https://github.com/hamidrezafahimi/Smart_POV.git) which funds basics for intelligence development on simulated voyager agents. The original code is editted such that it simply interacts - and may be evaluated - with our developed professional (D)RL-evaluation platform [MIIO2V](https://github.com/mohammadr-kaz/MIIO2V.git). A comparison of this project with [other similar works]() id documented [here]().


# PPO
Proximal Policy Optimiation

A package to train and evaluate an agent in order to reach a target defined in a 2D environment created from reality.

## Prerequisites

**- OpenCV**

*VERSION: at least 3.4.0*


**- Tensorflow**

*VERSION: 1.6*

Also works with tensorflow 2. So if you have the version 2 use the following lines of code:

```
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

```
instead of:

```
imoprt tensorflow as tf

```
## Run

In order to run in traning mode, set LOAD variable to False (ppo.py). In opposite if you want to run the code in evaluation mode, set LOAD variable to True then run this command in terminal:
```
python3 main.py

```
