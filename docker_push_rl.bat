@echo off

REM Set variables
set IMAGE_NAME=drawing_bot_rl
set IMAGE_TAG=latest
set REGISTRY=docker.io
set REPOSITORY=ho4040/drawing_bot

REM Tag the image for RunPod registry
docker tag %IMAGE_NAME%:%IMAGE_TAG% %REGISTRY%/%REPOSITORY%/%IMAGE_NAME%:%IMAGE_TAG%

REM Push the image to RunPod registry
docker push %REGISTRY%/%REPOSITORY%/%IMAGE_NAME%:%IMAGE_TAG% 