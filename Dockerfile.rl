# Use our base image with pre-downloaded VGG19 weights
FROM drawing_bot_base:latest

# Set environment variables
ENV TENSORBOARD_PORT=6007

# Copy source code
COPY . .

# Create necessary directories
RUN mkdir -p ./temp/rl/ckpt ./temp/rl/log

# Create startup script
RUN printf '#!/bin/bash\ntensorboard --logdir=/workspace/rl/log --port ${TENSORBOARD_PORT} --bind_all &\nexec "$@"' > /workspace/start.sh && \
    chmod +x /workspace/start.sh

# Expose Tensorboard port
EXPOSE ${TENSORBOARD_PORT}

# Set the default command
CMD ["/workspace/start.sh", "python3", "rl.py"]
