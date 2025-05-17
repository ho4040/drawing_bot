@echo off

REM Build RL image
docker build -t drawing_bot_rl:latest -f Dockerfile.rl . 