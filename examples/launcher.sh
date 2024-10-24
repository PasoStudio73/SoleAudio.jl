#!/bin/bash

N_THREADS=30

# ------------------------------------------------------------------------------------ #
#                                     itadata2024                                      #
# ------------------------------------------------------------------------------------ #
n=1
session="session_"$n
tmux new-session -d -s $session
tmux send-keys -t $session 'julia -i -t '$N_THREADS' ravdess_emotion_2bins.jl ' Enter
[ -n "${TMUX:-}" ]

n=2
session="session_"$n
tmux new-session -d -s $session
tmux send-keys -t $session 'julia -i -t '$N_THREADS' ravdess_emotion_8bins.jl ' Enter
[ -n "${TMUX:-}" ]

# n=3
# session="session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' respiratory_bronchiectasis.jl ' Enter
# [ -n "${TMUX:-}" ]

# n=4
# session="session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' respiratory_bronchiolitis.jl ' Enter
# [ -n "${TMUX:-}" ]

# n=5
# session="session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' respiratory_copd.jl ' Enter
# [ -n "${TMUX:-}" ]

# n=6
# session="session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' respiratory_pneumonia.jl ' Enter
# [ -n "${TMUX:-}" ]

# n=7
# session="session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' respiratory_urti.jl ' Enter
# [ -n "${TMUX:-}" ]