# Use Debian as base image
FROM ubuntu:latest

# Set working directory
WORKDIR /app

# Copy your project files into the container
COPY . .

RUN apt-get update

RUN apt-get install -y python3 python3-pip && \
    rm -rf /var/lib/apt/lists/*

# Set python3 as the default python interpreter
RUN ln -s /usr/bin/python3 /usr/bin/python

# Cache Dependencies
RUN pip3 install flake8
RUN pip3 install -r docs/requirements.txt

# Run Flake8
RUN flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
RUN flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

RUN ls docs/scripts

# Build docs
RUN bash docs/scripts/build.sh

# Define the default command to run when the container starts
CMD ["bash"]