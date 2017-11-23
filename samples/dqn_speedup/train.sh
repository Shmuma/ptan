#!/usr/bin/env bash
for seed in `seq 0 10`; do
    timeout 1h ./02_play_steps.py --cuda -s 1 --seed $seed
    timeout 1h ./02_play_steps.py --cuda -s 2 --seed $seed
    timeout 1h ./02_play_steps.py --cuda -s 3 --seed $seed
    timeout 1h ./02_play_steps.py --cuda -s 4 --seed $seed
    timeout 1h ./02_play_steps.py --cuda -s 5 --seed $seed
    timeout 1h ./02_play_steps.py --cuda -s 6 --seed $seed
done
