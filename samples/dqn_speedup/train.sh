#!/usr/bin/env bash
while true; do
    ./01_original.py --cuda
    ./02_play_steps.py --cuda -s 2
    ./02_play_steps.py --cuda -s 3
    ./02_play_steps.py --cuda -s 4
done
