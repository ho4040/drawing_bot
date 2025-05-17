@echo off

REM Set variables
set IMAGE_NAME=drawing_bot_bc
set IMAGE_TAG=latest
set REGISTRY=docker.io
set REPOSITORY=ho4040

REM Login to Docker Hub (if not already logged in)
REM docker login

REM Tag the image for Docker Hub
docker tag %IMAGE_NAME%:%IMAGE_TAG% %REPOSITORY%/%IMAGE_NAME%:%IMAGE_TAG%

REM Push the image to Docker Hub
docker push %REPOSITORY%/%IMAGE_NAME%:%IMAGE_TAG% 