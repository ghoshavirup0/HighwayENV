
A collection of environments for *autonomous driving* and tactical decision-making tasks

## The environments

### Highway

```python
env = gym.make("highway-v0")
```

In this task, the ego-vehicle is driving on a multilane highway populated with other vehicles.
The agent's objective is to reach a high speed while avoiding collisions with neighbouring vehicles. Driving on the right side of the road is also rewarded.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/highway.gif?raw=true"><br/>
    <em>The highway-v0 environment.</em>
</p>

A faster variant, `highway-fast-v0` is also available, with a degraded simulation accuracy to improve speed for large-scale training.

### Merge

```python
env = gym.make("merge-v0")
```

In this task, the ego-vehicle starts on a main highway but soon approaches a road junction with incoming vehicles on the access ramp. The agent's objective is now to maintain a high speed while making room for the vehicles so that they can safely merge in the traffic.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/merge-env.gif?raw=true"><br/>
    <em>The merge-v0 environment.</em>
</p>

### Roundabout

```python
env = gym.make("roundabout-v0")
```

In this task, the ego-vehicle if approaching a roundabout with flowing traffic. It will follow its planned route automatically, but has to handle lane changes and longitudinal control to pass the roundabout as fast as possible while avoiding collisions.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/roundabout-env.gif?raw=true"><br/>
    <em>The roundabout-v0 environment.</em>
</p>

### Parking

```python
env = gym.make("parking-v0")
```

A goal-conditioned continuous control task in which the ego-vehicle must park in a given space with the appropriate heading.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/parking-env.gif?raw=true"><br/>
    <em>The parking-v0 environment.</em>
</p>

### Intersection

```python
env = gym.make("intersection-v0")
```

An intersection negotiation task with dense traffic.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/intersection-env.gif?raw=true"><br/>
    <em>The intersection-v0 environment.</em>
</p>

### Racetrack

```python
env = gym.make("racetrack-v0")
```

A continuous control task involving lane-keeping and obstacle avoidance.

<p align="center">
    <img src="https://raw.githubusercontent.com/eleurent/highway-env/master/../gh-media/docs/media/racetrack-env.gif?raw=true"><br/>
    <em>The racetrack-v0 environment.</em>
</p>

## How to run the code??

Clone the repository in your local machine or cluster and install the following dependencies:
<ul>
<li>gym
<li>numpy
<li>pygame
<li>matplotlib
<li>pandas
<li>scipy
</ul>

<strong>Note:</strong> The code works with python version <strong>3.8</strong>. It is better to use <em>custom <strong>conda</strong> environment</em> to execute the program.
```
conda create --name custom_env python=3.8
```
All the dependencies with their corrosponding proper versions are mentioned in the <a href="requirements.txt" download>requirements.txt</a> file. Install all the dependencies by the following command in terminal
```
pip3 install -r requirements.txt
```



## Usage

Create a script in the <em>root</em> folder of the downloaded repository. 

<strong>Sample Script</strong>

```python
# Importing the libraries
import gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import highway_env


if __name__ == '__main__':

    # Selecting/Creating the environment
    env = gym.make('highway-v0')
    env = env.reset()
    
    # Training the model using Deep Q Network
    model = DQN('CnnPolicy', DummyVecEnv([env]),
                learning_rate=5e-4,
                buffer_size=15000,
                learning_starts=200,
                batch_size=32,
                gamma=0.8,
                train_freq=1,
                gradient_steps=1,
                target_update_interval=50,
                exploration_fraction=0.7,
                verbose=1,
                tensorboard_log="traied_cnn/")
    model.learn(total_timesteps=int(1e5))
    
    # Saving the trained model
    model.save("traied_cnn/model")

    
    # Loading the trained model
    model = DQN.load("model")

    # Rendering the model
    env = gym.make('highway-v0')
    obs = env.reset()
    done = False
    while not done:
        action,_=model.predict(obs)
        obs, _, done, _ = env.step(action)
        env.render()
    

```

The sample script should give output like the following


https://user-images.githubusercontent.com/57269014/164772662-93096bc2-003f-4efa-8d7e-0af95bd99eeb.mov



## Citing

If you use the project in your work, please consider citing it with:
```bibtex
@misc{highway-env,
  author = {Leurent, Edouard},
  title = {An Environment for Autonomous Driving Decision-Making},
  year = {2018},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/eleurent/highway-env}},
}
```

List of publications & preprints using `highway-env` (please open a pull request to add missing entries):
*   [Approximate Robust Control of Uncertain Dynamical Systems](https://arxiv.org/abs/1903.00220) (Dec 2018)
*   [Interval Prediction for Continuous-Time Systems with Parametric Uncertainties](https://arxiv.org/abs/1904.04727) (Apr 2019)
*   [Practical Open-Loop Optimistic Planning](https://arxiv.org/abs/1904.04700) (Apr 2019)
*   [α^α-Rank: Practically Scaling α-Rank through Stochastic Optimisation](https://arxiv.org/abs/1909.11628) (Sep 2019)
*   [Social Attention for Autonomous Decision-Making in Dense Traffic](https://arxiv.org/abs/1911.12250) (Nov 2019)
*   [Budgeted Reinforcement Learning in Continuous State Space](http://papers.nips.cc/paper/9128-budgeted-reinforcement-learning-in-continuous-state-space/) (Dec 2019)
*   [Multi-View Reinforcement Learning](http://papers.nips.cc/paper/8422-multi-view-reinforcement-learning) (Dec 2019)
*   [Reinforcement learning for Dialogue Systems optimization with user adaptation](https://tel.archives-ouvertes.fr/tel-02422691/) (Dec 2019)
*   [Distributional Soft Actor Critic for Risk Sensitive Learning](https://arxiv.org/abs/2004.14547) (Apr 2020)
*   [Bi-Level Actor-Critic for Multi-Agent Coordination](https://ojs.aaai.org/index.php/AAAI/article/view/6226) (Apr 2020)
*   [Task-Agnostic Online Reinforcement Learning with an Infinite Mixture of Gaussian Processes](https://arxiv.org/abs/2006.11441) (Jun 2020)
*   [Beyond Prioritized Replay: Sampling States in Model-Based RL via Simulated Priorities](https://arxiv.org/abs/2007.09569) (Jul 2020)
*   [Robust-Adaptive Interval Predictive Control for Linear Uncertain Systems](https://arxiv.org/abs/2007.10401) (Jul 2020)
*   [SMART: Simultaneous Multi-Agent Recurrent Trajectory Prediction](https://arxiv.org/abs/2007.13078) (Jul 2020)
*   [Delay-Aware Multi-Agent Reinforcement Learning for Cooperative and Competitive Environments](https://arxiv.org/abs/2005.05441) (Aug 2020)
*   [B-GAP: Behavior-Guided Action Prediction for Autonomous Navigation](https://arxiv.org/abs/2011.03748) (Nov 2020)
*   [Model-based Reinforcement Learning from Signal Temporal Logic Specifications](https://arxiv.org/abs/2011.04950) (Nov 2020)
*   [Robust-Adaptive Control of Linear Systems: beyond Quadratic Costs](https://arxiv.org/abs/2002.10816) (Dec 2020)
*   [Assessing and Accelerating Coverage in Deep Reinforcement Learning](https://arxiv.org/abs/2012.00724) (Dec 2020)
*   [Distributionally Consistent Simulation of Naturalistic Driving Environment for Autonomous Vehicle Testing](https://arxiv.org/abs/2101.02828) (Jan 2021)
*   [Interpretable Policy Specification and Synthesis through Natural Language and RL](https://arxiv.org/abs/2101.07140) (Jan 2021)
*   [Deep Reinforcement Learning Techniques in Diversified Domains: A Survey](https://link.springer.com/article/10.1007/s11831-021-09552-3) (Feb 2021)
*   [Corner Case Generation and Analysis for Safety Assessment of Autonomous Vehicles](https://arxiv.org/abs/2102.03483) (Feb 2021)
*   [Intelligent driving intelligence test for autonomous vehicles with naturalistic and adversarial environment](https://www.nature.com/articles/s41467-021-21007-8) (Feb 2021)
*   [Building Safer Autonomous Agents by Leveraging Risky Driving Behavior Knowledge](https://arxiv.org/abs/2103.10245)
*   [Quick Learner Automated Vehicle Adapting its Roadmanship to Varying Traffic Cultures with Meta Reinforcement Learning](https://arxiv.org/abs/2104.08876) (Apr 2021)
*   [Deep Multi-agent Reinforcement Learning for Highway On-Ramp Merging in Mixed Traffic](https://arxiv.org/abs/2105.05701) (May 2021)
*   [Accelerated Policy Evaluation: Learning Adversarial Environments with Adaptive Importance Sampling](https://arxiv.org/abs/2106.10566) (Jun 2021)
*   [Learning Interaction-aware Guidance Policies for Motion Planning in Dense Traffic Scenarios](https://arxiv.org/abs/2107.04538) (Jul 2021)
*   [Robust Predictable Control](https://arxiv.org/abs/2109.03214) (Sep 2021)
