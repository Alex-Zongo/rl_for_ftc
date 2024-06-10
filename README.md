# Fault-Tolerant Attitude Flight Control via Reinforcement Learning #

[![Python](https://shields.io/badge/python-v3.10-blue.svg)](mygithubrepo)
[![License]()](mygithubrepo)

## Author
👤 **Alex Zongo**
(Tsinghua University MSc. Eng. Student)
* Github: [@AZongo](https://github.com/Alex-Zongo)
* LinkedIn: [![@AZongo](https://shields.io/badge/LinkedIn--blue?style=social&logo=linkedin)](https://www.linkedin.com/in/alex-zongo/)

> This project is an integral part of my master thesis at the Navigation and Control Lab in the Department of Automation at Tsinghua University


The overall idea is to use reinforcement learning strategies (including improvements) for smooth fault-tolerant flight control. Smoothness here is defined as smooth sequence of control inputs(actions - RL terminolgy).

Two main algorithms are investigated: Proximal Policy Optimization (PPO) and Cross-Entropy Methods - Twin Delayed Deep Deterministic Policy Gradient (CEM-TD3).
They are applied on a high-fidelity fixed-wing aircraft trimmed at specific flight conditions.

> - available environment configurations: nominal (H=2km,V=90m/s), low-q (H=10km, V=90m/s), high-q (H=2km,V=150m/s), gust (external disturbance of 15ft/s up-pointing  wind gust), cg (CG aft-shifted), be (broken elevator), jr (jammed rudder), sa (saturated aileron), se (saturated elevator), ice (ice on wings)


## Paper Abstract
This study advances flight control systems by integrating deep reinforcement learning to enhance fault tolerance in fixed-wing air- craft. We assess the efficiency of Cross-Entropy Method Reinforcement Learning (CEM-RL) and Proximal Policy Optimization (PPO) algo- rithms in developing an adaptive stable attitude controller. Our proposed frameworks, focusing on smooth actuator control, showcase improved robustness across standard and fault-induced scenarios. The algorithms demonstrate unique traits in terms of trade-offs between trajectory track- ing and control smoothness. Our approach that results in state-of-the-art performance with respect to benchmarks, presents a leap forward in autonomous aviation safety.

## Module Installation Instructions
The module is best suited for Linux.
1. Create and activate a python virtual environment in the project root:
``` python3.10 -m venv .venv```

2. Install the required packages with for instance (optional):
```python3.10 -m pip install -r requirements.txt```

## CEM-TD3 TRAINING and EVAL
* For training: Check the file "ES_TD3Buffers.py" for additional options and descriptions.
run ```python3 ES_TD3Buffers.py --seed=0 --elitism --cem-with-adapt```

* For evaluation: Check the file "evaluate_es.py" for additional options and descriptions.
run ```python3 evaluate_es.py --use-best-mu --env-name=$ENV-NAME$ --agent-name=$your-trained-agent-name$```

To compare the behavior of your agent on 2 different environments or flight regimes:
you can run ```python3 evaluate_es.py --use-best-mu --env-name=$YOUR-ENV$ --env2-name=$YOUR-SECOND-ENV$ --agent-name=your-trained-agent-name```

Look into the file "online_adaptation.py" for details regarding the evaluation of multiple trained agents in an adaptation scheme based on the environment via an identification mechanism.
e.g. ```python3 online_adaptation.py --env-name=$YOUR-ENV$ --t-max=80 --use-second-env --env2-name=$YOUR-OTHER-ENV$ --switch-time=30```


## PPO TRAINING and EVAL
* For training: see "ppo_continuous_actions.py" file for arguments description.
you can run ```python3 ppo_continuous_actions.py --gym-id=$ENV-NAME$  --seed=$YOUR-SEED-NUMBER$ --total-timesteps=$TRAINING-TOTAL_TIMESTEPS$ --cuda```

* For evaluation: see "ppo_continuous_eval.py" for details on the arguments and options
you can run:
```python3 ppo_continuous_eval.py --gym-id=$YOUR-ENV$ --agent-name=$YOUR-TRAINED-AGENT-NAME.pkl$ --t-max=$MAX-EVAL-TIME$ --amp-theta=$[list of reference pitch angle]$ --amp-phi=$[list of reference roll angle]$```

If you are considering fault onset at different stages of the flight evaluation you can add the following options to the above script
```--use-second-env --gym2-id=YOUR-SECOND-ENV --switch-time=TIMESTEP-TO-SWITCH-TO-SECOND-ENV```


* **A Deep Neural Network FDI was trained on different cases or scenarios, following this project to identify the presence of faults**. The reason is to evaluate and compare the barebone ppo controller with the case where a FDI is present. A filter is then trained to smoothly integrate the FDI to the controller, via the script in the file "fdi_adaptation2.py".


* **Evaluating the performance of a trained agent, on different trajectories**. The file "trajectory_comp.py" comprehensively describes the process.
```python3 trajectory_comp.py --gym-id=$YOUR-ENV$ --agent-name=$PPO-TRAINED-AGENT$```

* The file "comparing_performance.py" is used to compartively evaluate the performance of the PPO agent with respect to a pair of different scenarios.
```python3 comparing_performance.py --gym-id=$ENV1$ --gym2-id=$ENV2$ --agent-name=$NAME.pkl$```


* **Stability Analyses**
    * for CEM-TD3 -> look into file "cemtd3_agent_stability.py". Running this file shall generate eigen-values plot on a C-plane as well as a time series attitude response through the given environment.
    e.g.: ```python3 cemtd3_agent_stability.py --agent-name=$CEMTD3_AGENT$ --use-best-mu --env-name=$ENV$```

    * for PPO -> look into file "ppo_agent_stability.py". It has similar structure with "cemtd3_agent_stability.py". Fill in trained agents in <<controllers_names>>
    ```python3 ppo_agent_stability.py --gym-id=$ENV$```


## Paper
A paper is presented at the International Conference on Guidance Navigation and Control (ICGNC) in Changsha, August 09-11 2024. Link to be shared soon!!

## References

The environment and some part of the code was borrowed from TU Delft University from the following works:

* **Vlad Gavra**, "Evolutionary Reinforcement Learning: A Hybrid Approach in Intelligent Fault-tolerant Flight Control Systems" [[thesis](https://bit.ly/3D7mj0i)].[[code](https://github.com/VladGavra98/SERL.git)]

* **Killian Dally and E.-J. Van Kampen** "Soft Actor-Critic Deep Reinforcement Learning for Fault Tolerant Flight Control" (2022)[doi.org/10.2514/6.2022-2078][[paper](https://doi.org/10.2514/6.2022-2078)]. [[code](https://github.com/kdally/fault-tolerant-flight-control-drl.git)]

* **Huang, Shengyi et al.** "The 37 Implementation Details of Proximal Policy Optimization" (2022) [[doc](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)] [[code](https://github.com/vwxyzjn/ppo-implementation-details.git)]

* **Olivier Pourchot** "CEM-RL" [[paper](https://arxiv.org/pdf/1810.01222.pdf)] [[code](https://github.com/apourchot/CEM-RL.git)]
