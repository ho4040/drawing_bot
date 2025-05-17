@echo off

REM Set variables
set IMAGE_NAME=drawing_bot_bc
set IMAGE_TAG=latest
set CONTAINER_NAME=drawing_bot_bc_container

REM Set training parameters
set NUM_SAMPLES=300
set BATCH_SIZE=32
set NUM_EPOCHS=30
set LEARNING_RATE=1e-4
set CURRICULUM_PHASES=10
set STROKES_INCREASE_EPOCHS=100

REM Set W&B parameters
set WANDB_API_KEY=your_api_key
set WANDB_PROJECT=drawing_bot
set WANDB_ENTITY=your_username
set WANDB_RUN_NAME=bc_training

REM Create temp directory if it doesn't exist
if not exist "temp" mkdir temp

REM Run the Docker container
docker run -it --rm ^
    --gpus all ^
    --name %CONTAINER_NAME% ^
    -v %cd%/temp:/workspace/temp ^
    -e WANDB_API_KEY=%WANDB_API_KEY% ^
    -e WANDB_PROJECT=%WANDB_PROJECT% ^
    -e WANDB_ENTITY=%WANDB_ENTITY% ^
    -e WANDB_RUN_NAME=%WANDB_RUN_NAME% ^
    %IMAGE_NAME%:%IMAGE_TAG% ^
    python3 bc.py ^
    --num_samples %NUM_SAMPLES% ^
    --batch_size %BATCH_SIZE% ^
    --num_epochs %NUM_EPOCHS% ^
    --learning_rate %LEARNING_RATE% ^
    --curriculum_phases %CURRICULUM_PHASES% ^
    --strokes_increase_epochs %STROKES_INCREASE_EPOCHS% ^
    --output_dir /workspace/temp/bc 