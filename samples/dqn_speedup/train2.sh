#!/usr/bin/env bash
while true; do
    timeout 1h ./03_parallel.py --cuda
    timeout 1h ./04_cuda_async.py --cuda
    timeout 1h ./05_new_wrappers.py --cuda
done
