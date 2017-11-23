#!/usr/bin/env bash
for s in `seq 0 100`; do
    timeout 10m ./05_new_wrappers.py --cuda --seed $s
done
