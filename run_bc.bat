@echo off

REM Set training parameters
set NUM_SAMPLES=300
set BATCH_SIZE=32
set NUM_EPOCHS=30
set LEARNING_RATE=1e-4
set CURRICULUM_PHASES=10
set STROKES_INCREASE_EPOCHS=100


REM Create temp directory if it doesn't exist
if not exist "temp" mkdir temp

python bc.py ^
    --num_samples %NUM_SAMPLES% ^
    --batch_size %BATCH_SIZE% ^
    --num_epochs %NUM_EPOCHS% ^
    --learning_rate %LEARNING_RATE% ^
    --curriculum_phases %CURRICULUM_PHASES% ^
    --strokes_increase_epochs %STROKES_INCREASE_EPOCHS% ^
    --output_dir ./temp/bc