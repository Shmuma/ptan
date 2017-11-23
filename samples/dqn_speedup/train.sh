#!/usr/bin/env bash
while true; do
    ./02_play_steps.py --cuda -s 1
    ./02_play_steps.py --cuda -s 2
    ./02_play_steps.py --cuda -s 3
    ./02_play_steps.py --cuda -s 4
    ./02_play_steps.py --cuda -s 5
    ./02_play_steps.py --cuda -s 6
done
