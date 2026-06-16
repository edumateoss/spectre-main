FROM mcr.microsoft.com/devcontainers/python:3.12-trixie

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
