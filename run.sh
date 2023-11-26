#!/bin/bash

if [ -z "$TENSORBOARD_DIR" ]; then
    TENSORBOARD_DIR=/workspace/runs
fi

# Start TensorBoard in the background and redirect its output to a log file for debugging.
tensorboard --logdir=$TENSORBOARD_DIR --port 6007 &> ./tensorboard.log &

# Wait for TensorBoard to start. This can be replaced with a more robust check.
sleep 5

# Run the training script.
python3 train.py

