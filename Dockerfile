FROM runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel


WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends libcairo2-dev pkg-config python3-dev && \
    rm -rf /var/lib/apt/lists/*


COPY requirements.txt .

RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the necessary source files.
COPY drawing_env.py train.py target.png run.sh ./

# Expose Tensorboard
EXPOSE 6007 


# Ensure the script is executable.
RUN chmod +x run.sh

# The ENTRYPOINT instruction is used to set the container to run as an executable.
# Changed CMD to ENTRYPOINT as it's more appropriate for initiating the run.
ENTRYPOINT ["/bin/bash", "run.sh"]
