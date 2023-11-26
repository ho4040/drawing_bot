docker run --gpus all -v ./temp:/workspace  -p 8888:6007 -e TOTAL_TIMESTEPS=40000 -e MAX_STEPS=50 -e DIFFICULTY_INC_PERIOD=1000 -e N_ENV=2 -e CHECK_FREQ=50 -e TORCH_HOME=/workspace/cache -e TENSORBOARD_DIR=/workspace/runs -e CHECKPOINT_PATH=/workspace/ckpt -e ALGORITHM=SAC -e SAC_BUFFER_SIZE=10000 drawing-bot-train
