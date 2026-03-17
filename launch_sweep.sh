#!/bin/bash

# Check if sweep ID is provided
if [ -z "$1" ]; then
    echo "Usage: ./launch_sweep.sh <sweep_id>"
    echo "Example: ./launch_sweep.sh ag42-iit-bombay/Profile_VAE/v4zmppj7"
    exit 1
fi

SWEEP_ID=$1
SESSION_NAME="wandb_sweep_$(date +%s)"

# Environment variables - changed PROJECT_DIR to the requested subfolder
PROJECT_DIR="/home/aarav/Profile_VAE/eb_jepa"
EBJEPA_DSETS="/data/TCGA_cleaned"
EBJEPA_CKPTS="checkpoints"

# Start a new tmux session in the background
tmux new-session -d -s "$SESSION_NAME"

# Pane 0: Start nvidia-smi immediately in the only existing pane
tmux send-keys -t "$SESSION_NAME.0" "nvidia-smi -l 1" C-m

# Split horizontally: Pane 0 stays TOP (nvidia-smi), Pane 1 becomes BOTTOM
tmux split-window -v -t "$SESSION_NAME.0"

# Agent starting command - removed quotes from values as requested
CMD="cd $PROJECT_DIR && conda activate multiomics_pvae && export EBJEPA_DSETS=$EBJEPA_DSETS && export EBJEPA_CKPTS=$EBJEPA_CKPTS && wandb agent $SWEEP_ID"

# Now we have Pane 0 (Top) and Pane 1 (Bottom)
# Start the first agent in Pane 1
tmux send-keys -t "$SESSION_NAME.1" "$CMD" C-m

# Split the bottom pane (1) horizontally 6 more times to get 7 panes in total
for i in {2..7}; do
    tmux split-window -h -t "$SESSION_NAME.1"
    # Evenly distribute space
    tmux select-layout -t "$SESSION_NAME" tiled
done

# Start agents in the newly created panes (2 to 7)
for i in {2..7}; do
    tmux send-keys -t "$SESSION_NAME.$i" "$CMD" C-m
done

# Attach to the session
echo "Launched sweep $SWEEP_ID in tmux session: $SESSION_NAME"
tmux attach-session -t "$SESSION_NAME"