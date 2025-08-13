#!/bin/zsh

python self_play.py --board-size 9 --learning-agent experiences/policy_agent_initial.hdf5 --num-games 32 --experience-out experiences/initial_20250813_1a.hdf5 > self_play_20250813_1a.out &
python self_play.py --board-size 9 --learning-agent experiences/policy_agent_initial.hdf5 --num-games 32 --experience-out experiences/initial_20250813_1b.hdf5 > self_play_20250813_1b.out &
