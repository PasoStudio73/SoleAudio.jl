#!/bin/bash

N_THREADS=30

# ------------------------------------------------------------------------------------ #
#                                        gender                                        #
# ------------------------------------------------------------------------------------ #
# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/gender_age"
# FILE_NAME="spcds_gender_full_modal"
# LABEL="propositional"
# n=1
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/gender_age"
# FILE_NAME="spcds_gender_full_modal"
# LABEL="modal"
# n=2 
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/gender_age"
# FILE_NAME="spcds_gender_full_modal"
# LABEL="multimodal"
# n=3
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# ------------------------------------------------------------------------------------ #
#                                      age 2 splits                                    #
# ------------------------------------------------------------------------------------ #
# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/gender_age"
# FILE_NAME="spcds_age2bins_full_modal"
# LABEL="propositional"
# n=4
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/gender_age"
# FILE_NAME="spcds_age2bins_full_modal"
# LABEL="modal"
# n=5
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/gender_age"
# FILE_NAME="spcds_age2bins_full_modal"
# LABEL="multimodal"
# n=6
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# ------------------------------------------------------------------------------------ #
#                                      age 4 splits                                    #
# ------------------------------------------------------------------------------------ #
# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/gender_age"
# FILE_NAME="spcds_age4bins_full_modal"
# LABEL="propositional"
# n=7
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/gender_age"
# FILE_NAME="spcds_age4bins_full_modal"
# LABEL="modal"
# n=8
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/gender_age"
# FILE_NAME="spcds_age4bins_full_modal"
# LABEL="multimodal"
# n=9
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# ------------------------------------------------------------------------------------ #
#                                     emotion 2 bins                                   #
# ------------------------------------------------------------------------------------ #
# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/emotion"
# FILE_NAME="emods_emo2bins_full_modal"
# LABEL="propositional"
# n=10
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/emotion"
# FILE_NAME="emods_emo2bins_full_modal"
# LABEL="modal"
# n=11
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/emotion"
# FILE_NAME="emods_emo2bins_full_modal"
# LABEL="multimodal"
# n=12
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# # ------------------------------------------------------------------------------------ #
# #                                     emotion 8 bins                                   #
# # ------------------------------------------------------------------------------------ #
# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/emotion"
# FILE_NAME="emods_emo8bins_full_modal"
# LABEL="propositional"
# n=13
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/emotion"
# FILE_NAME="emods_emo8bins_full_modal"
# LABEL="modal"
# n=14
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/emotion"
# FILE_NAME="emods_emo8bins_full_modal"
# LABEL="multimodal"
# n=15
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# ------------------------------------------------------------------------------------ #
#                                    healty vs pneuma                                  #
# ------------------------------------------------------------------------------------ #
# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/respiratory"
# FILE_NAME="respiratory_Healthy_Pneumonia"
# LABEL="propositional"
# n=16
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/respiratory"
# FILE_NAME="respiratory_Healthy_Pneumonia"
# LABEL="modal"
# n=17
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/respiratory"
# FILE_NAME="respiratory_Healthy_Pneumonia"
# LABEL="multimodal"
# n=18
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# ------------------------------------------------------------------------------------ #
#                                healty vs Bronchiectasis                              #
# ------------------------------------------------------------------------------------ #
# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/respiratory"
# FILE_NAME="respiratory_Healthy_Bronchiectasis"
# LABEL="propositional"
# n=19
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/respiratory"
# FILE_NAME="respiratory_Healthy_Bronchiectasis"
# LABEL="modal"
# n=20
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/respiratory"
# FILE_NAME="respiratory_Healthy_Bronchiectasis"
# LABEL="multimodal"
# n=21
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# # ------------------------------------------------------------------------------------ #
# #                                healty vs Bronchiolitis                               #
# # ------------------------------------------------------------------------------------ #
# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/respiratory"
# FILE_NAME="respiratory_Healthy_Bronchiolitis"
# LABEL="propositional"
# n=22
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/respiratory"
# FILE_NAME="respiratory_Healthy_Bronchiolitis"
# LABEL="modal"
# n=23
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/respiratory"
# FILE_NAME="respiratory_Healthy_Bronchiolitis"
# LABEL="multimodal"
# n=24
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# # ------------------------------------------------------------------------------------ #
# #                                     healty vs copd                                   #
# # ------------------------------------------------------------------------------------ #
# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/respiratory"
# FILE_NAME="respiratory_Healthy_COPD"
# LABEL="propositional"
# n=25
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/respiratory"
# FILE_NAME="respiratory_Healthy_COPD"
# LABEL="modal"
# n=26
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/respiratory"
# FILE_NAME="respiratory_Healthy_COPD"
# LABEL="multimodal"
# n=27
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# # ------------------------------------------------------------------------------------ #
# #                                     healty vs urti                                   #
# # ------------------------------------------------------------------------------------ #
# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/respiratory"
# FILE_NAME="respiratory_Healthy_URTI"
# LABEL="propositional"
# n=28
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/respiratory"
# FILE_NAME="respiratory_Healthy_URTI"
# LABEL="modal"
# n=29
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# DATASET_ORIGIN="Mozilla"
# FILE_PATH="/home/paso/datasets/Speech/respiratory"
# FILE_NAME="respiratory_Healthy_URTI"
# LABEL="multimodal"
# n=30
# session="Spc_session_"$n
# tmux new-session -d -s $session
# tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
# [ -n "${TMUX:-}" ]

# ------------------------------------------------------------------------------------ #
#                                   respiratory 4 bins                                 #
# ------------------------------------------------------------------------------------ #
DATASET_ORIGIN="Mozilla"
FILE_PATH="/home/paso/datasets/Speech/respiratory"
FILE_NAME="respiratory_Healthy_URTI_COPD_Pneumonia"
LABEL="propositional"
n=31
session="Spc_session_"$n
tmux new-session -d -s $session
tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
[ -n "${TMUX:-}" ]

DATASET_ORIGIN="Mozilla"
FILE_PATH="/home/paso/datasets/Speech/respiratory"
FILE_NAME="respiratory_Healthy_URTI_COPD_Pneumonia"
LABEL="modal"
n=232
session="Spc_session_"$n
tmux new-session -d -s $session
tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
[ -n "${TMUX:-}" ]

DATASET_ORIGIN="Mozilla"
FILE_PATH="/home/paso/datasets/Speech/respiratory"
FILE_NAME="respiratory_Healthy_URTI_COPD_Pneumonia"
LABEL="multimodal"
n=33
session="Spc_session_"$n
tmux new-session -d -s $session
tmux send-keys -t $session 'julia -i -t '$N_THREADS' sspeech.jl '$DATASET_ORIGIN' '$FILE_NAME' '$FILE_PATH' '$LABEL' 2>&1 | tee '$FILE_NAME-$LABEL'.out &' Enter
[ -n "${TMUX:-}" ]