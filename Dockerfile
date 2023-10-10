# Use Debian as base image
FROM debian:latest

# Set working directory
WORKDIR /app

# Install necessary packages
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Copy your project files into the container
COPY . .

# Cache Dependencies
RUN pip3 install --upgrade pip
RUN pip3 install flake8
RUN pip3 install -r docs/requirements.txt

# Run Flake8
RUN flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
RUN flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Build docs
RUN bash docs/scripts/build.sh

# Define the default command to run when the container starts
CMD ["bash"]