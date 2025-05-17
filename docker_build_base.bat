@echo off

REM Build base image
docker build -t drawing_bot_base:latest -f Dockerfile.base . 