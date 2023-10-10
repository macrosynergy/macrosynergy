# Use Debian as base image
FROM python:3.8

# Set the working directory
WORKDIR /app

RUN pip3 install flake8

# Copy only the requirements file to avoid unnecessary cache invalidation
COPY ./docs/requirements.txt .

# Install the dependencies (if requirements.txt hasn't changed, this step will use the cache)
RUN pip install -r requirements.txt

# Copy the rest of the application files
COPY . .

# Run Flake8
RUN flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
RUN flake8 . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

# Build docs
RUN bash docs/scripts/build.sh

# Define the default command to run when the container starts
CMD ["bash"]