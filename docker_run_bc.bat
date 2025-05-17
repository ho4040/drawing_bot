@echo off
setlocal enabledelayedexpansion

REM .env 파일에서 WANDB_API_KEY 읽기
for /f "tokens=*" %%a in (.env) do (
    set "%%a"
)

REM Docker 컨테이너 실행
docker run --gpus all ^
    -v %cd%:/workspace ^
    -e WANDB_API_KEY=%WANDB_API_KEY% ^
    drawing_bot_bc:latest ^
    python bc.py %* 