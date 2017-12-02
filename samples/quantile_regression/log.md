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

2e3e814acb527ce0237d96b75f22f3091468fb47, run Dec02_13-11-56_gpu-pong-qr-qr-paper
Not converging, tails are huge, Q-values are wrong

Try to fix tau_hat shape:
9053441ffb7b2f163f58647fa586e052830c22ce, run Dec02_13-23-41_gpu-pong-qr-qr-paper-2
No convergence

## Quantile regression fixed

Implement QR with penalization fixed: loss is scaled with tau - dirac{u > 0}

e9f811f809cc9498bd69dc56446b4a22a4bbddcc, run Dec02_13-13-51_gpu-pong-qr-qr-fixed
Not converging, tails are huge, Q-values are wrong.

Try to fix tau_hat shape:
ed272209e40b92c2ca74086e96b9d494cbfba797, run Dec02_13-21-55_gpu-pong-qr-qr-fixed-2
No convergence

## Sort of tau_hat

Try to sort tau_hat by quantile values

1. fixed version, ascending sort: ea685802aaa002558822f451a477024acbaef1cc, Dec02_14-54-23_gpu-pong-qr-sort-fixed-asc
2. fixed version, descending sort: 82235af9d2ca74228d756ca1c8c5ed0044c6c61b, Dec02_14-55-13_gpu-pong-qr-sort-fixed-desc 

Both are diverging.

1. orig version, ascending sort: 
2. orig version, descending sort: 
