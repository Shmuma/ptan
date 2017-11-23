#!/usr/bin/env bash
for s in `seq 0 100`; do
    timeout 10m ./02_play_steps.py --cuda --steps 2 --seed $s
done
