#!/usr/bin/env bash
while true; do
    ./03_parallel.py --cuda -s 1
    ./03_parallel.py --cuda -s 2
    ./03_parallel.py --cuda -s 3
    ./04_cuda_async.py --cuda -s 1
    ./04_cuda_async.py --cuda -s 2
    ./04_cuda_async.py --cuda -s 3
    ./05_new_wrappers.py --cuda -s 1
    ./05_new_wrappers.py --cuda -s 2
    ./05_new_wrappers.py --cuda -s 3
done
