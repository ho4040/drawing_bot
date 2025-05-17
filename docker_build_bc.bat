@echo off

REM Set variables
set IMAGE_NAME=drawing_bot_bc
set IMAGE_TAG=latest

REM Build the Docker image
docker build -t %IMAGE_NAME%:%IMAGE_TAG% -f Dockerfile.bc . 