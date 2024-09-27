# WISMAQ
Adaptive Policy Regularization for Offline-to-Online Reinforcement Learning via Weighted Increased Simple Moving Average Q-value

## Offline-to-Online Reinforcement Learning
Offline-to-online (O2O) RL follows the same assumption as offline
RL, i.e., there is no access to the simulator of the system. However,
we could further improve the model with online interactions since
the pure offline method cannot yield accurate value estimation of
the OOD state-action values. Hence, the goal is to enhance the
capability of the model with online training without learning from
scratch as in the traditional online setting.

To our knowledge, we are the first to study offline-to-online
reinforcement learning for building control systems. Prior methods
developed for other domains need at least one of the following
requirements that makes them resource-consuming and most of them fail to maintain the pre-trained performance
in building environments:

**R.1** Require information on absolute scores of expert and random
agents, typically not be available for buildings.

**R.2** Many suffer policy collapse at the very beginning of the
transition from offline mode to online mode.

**R.3** Introduce compute overhead with additional models and/or
replay buffers.

To summarize, the key contributions of our work are:
-  A fine-tuning based offline-to-online RL algorithm that maintains
the pre-trained model’s ability and continues to improve
with online interactions (to tackle R.1 and R.2).

-  The add-on methods - Combined Experience Replay (CER)
and Bootstrapped Ensemble help adapt the distribution
drift with extreme low cost of O(1) time complexity.

-  Our method requires no extra models, only a single replay
buffer and works without differentiating the offline and online
transitions (to tackle R.3).

## Methodology - Weighted Increased Simple Moving Average Q-value
WISMAQ optimizes the policy by identifying
if the current SMA of the averaged Q-value is higher than a
previous reference SMA. If so, the WISMAQ loss term is activated,
it encourages the policy to explore. Otherwise, when the WISMAQ
loss term is lower than the reference value, we keep the actor loss
as is since it means the Q-value is converging, i.e., leave it learning
as the original offline model, which is conservatively trained with
online transitions.

## How to run the codes
0. Install required libraries: pytorch, Sinergym, etc.
1. Modify the parameters and paths in the main_off2online.py
2. Execute ```python main_off2online.py```
