#!/bin/bash

python main.py --model_dir=weight_01 \
        --max_timesteps=15 --max_episodes_num=1000 \
        --start_episodes=100 --eval_freq_episode=50 \
        --batch_size=1024 --discount=0.99 \
        --victim_n_episodes=60 --ae_n_epochs=1 \
        --weight=0.1 \