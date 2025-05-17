@echo off

REM Set variables
set IMAGE_NAME=drawing_bot_rl
set IMAGE_TAG=latest
set CONTAINER_NAME=drawing_bot_rl_container

REM Set training parameters
set NUM_ENV=4
set MAX_STEPS=50
set TOTAL_TIMESTEPS=200000
set CHECK_FREQ=5000
set TENSORBOARD_DIR=./temp/rl/log
set TORCH_HOME=./temp/cache
set CHECKPOINT_PATH=./temp/rl/ckpt
set ALGORITHM=PPO

REM Create temp directory if it doesn't exist
if not exist "temp" mkdir temp

REM Run the Docker container
docker run -it --rm ^
    --gpus all ^
    --name %CONTAINER_NAME% ^
    -v %cd%/temp:/workspace/temp ^
    -p 6007:6007 ^
    -e NUM_ENV=%NUM_ENV% ^
    -e MAX_STEPS=%MAX_STEPS% ^
    -e TOTAL_TIMESTEPS=%TOTAL_TIMESTEPS% ^
    -e CHECK_FREQ=%CHECK_FREQ% ^
    -e TENSORBOARD_DIR=%TENSORBOARD_DIR% ^
    -e TORCH_HOME=%TORCH_HOME% ^
    -e CHECKPOINT_PATH=%CHECKPOINT_PATH% ^
    -e ALGORITHM=%ALGORITHM% ^
    %IMAGE_NAME%:%IMAGE_TAG% 