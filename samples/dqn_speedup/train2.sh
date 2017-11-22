#!/usr/bin/env bash
while true; do
    ./03_parallel.py --cuda
    ./04_cuda_async.py --cuda
    ./05_new_wrappers.py --cuda
done
