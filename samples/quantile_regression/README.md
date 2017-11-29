# Quantile Regression

Implementation of Quantile Regression method from the article 
["Distributional Reinforcement Learning with Quantile Regression"](https://arxiv.org/abs/1710.10044)  

Nervana implementation in coach:
1. loss https://github.com/NervanaSystems/coach/blob/master/architectures/tensorflow_components/heads.py
2. agent https://github.com/NervanaSystems/coach/blob/master/agents/qr_dqn_agent.py

## Open questions

### Sorting of output quantile's values

In coach, they sort network output by value before calculating loss. Why?

### Dirac on loss

Coach has a bug in loss calculation: https://github.com/NervanaSystems/coach/issues/29

I've tried to fix it in my version, but result is weird -- first and last quantilles becomes 
very by magnitude, which leads to bad expected Q values. Version which doesn't have diracs subtracted
converges fine, but that's weird. 

### Logic error in the article

In the article they say that QR loss is penalizes overestimation errors with weight tau 
and underestimation errors with weight 1-tau. But **overestimation** means that the predicted value
is larger than target, and vice versa. 

Later, in the Algorithm 1 they calculate quantile Huber loss from projected update (target) minus predicted
by the network.

But in quantile Huber loss Dirac delta is passed the argument of u < 0, which means that we penalize 
with opposite weights.
