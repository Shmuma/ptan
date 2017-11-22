#!/usr/bin/env bash
while true; do
    timeout 1h ./01_original.py --cuda
    timeout 1h ./02_play_steps.py --cuda -s 2
    timeout 1h ./02_play_steps.py --cuda -s 3
    timeout 1h ./02_play_steps.py --cuda -s 4
done
