FROM python:3.10.10-bullseye as build

# Add SSH keys
RUN mkdir -p -m 0700 ~/.ssh && ssh-keyscan github.com >> ~/.ssh/known_hosts

#WORKDIR /app

# Install requirements
# https://pythonspeed.com/articles/docker-cache-pip-downloads/
COPY requirements.txt requirements.txt
RUN --mount=type=ssh --mount=type=cache,target=~/Library/Caches/pip \
    pip install -r requirements.txt

# Copy source code
COPY src src
COPY setup.cfg setup.cfg
COPY pyproject.toml pyproject.toml

# Install rifs
RUN --mount=type=ssh --mount=type=cache,target=~/Library/Caches/pip \
     pip install --target=/pip-packages .

# Clean up SSH keys
RUN rm -rf /root/.ssh/

# PUBLISH IMAGE
FROM python:3.10.10-slim-bullseye as publish

# Install git and ffmpeg
RUN apt-get update && \
    apt-get install -y git && \
    apt-get install -y ffmpeg

#WORKDIR /app

# Copy rifs
COPY --from=build /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages

COPY --from=build /usr/local/bin /usr/local/bin

# Make sure scripts in .local are usable:
ENV PATH=/root/.local/bin:$PATH

# Entrypoint can be overridden with other arguments
ENTRYPOINT ["rifs"]

