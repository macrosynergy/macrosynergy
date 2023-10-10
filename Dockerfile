# Use Debian as base image
FROM python:3

# Set working directory
WORKDIR /app

# Copy your project files into the container
COPY . .

# Cache Dependencies
RUN pip3 install flake8
RUN pip3 install -r docs/requirements.txt

# Run Flake8
RUN flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
RUN flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

RUN ls

# Build docs
RUN bash docs/scripts/build.sh

# Define the default command to run when the container starts
CMD ["bash"]