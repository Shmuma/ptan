# 2017-12-02

Experiments with loss

## MSE
Raw mse loss: 71297f56d8a2dc3aa80b986effd7e5be854ebc18, run Dec02_12-31-57_gpu-pong-qr-mse

Converge, but not very fast

## Huber loss without multipliers

c3620ebf8ab81fe5bfbb5502f9419f72614d5bf3, run Dec02_12-46-13_gpu-pong-qr-huber

Convergence is better.

## Quantile regression as in paper

Implement quantile regression as in paper: loss is scaled with tau-dirac{u < 0}
