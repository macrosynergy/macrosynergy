# Use Python 3.7 Slim Bookworm as the base image
FROM python:3.7.9

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory
WORKDIR /app

# Upgrade pip, setuptools, and wheel
RUN pip install --upgrade pip setuptools wheel

# Install Cython
RUN pip install cython==0.29

# RUN pip install numpy==1.16.6
RUN pip install numpy==1.21.1

WORKDIR /app

# Build and install Pandas
RUN pip install pandas==0.24.2 --no-deps

# Install additional Python dependencies
RUN pip install pytz python-dateutil 
# statsmodels==0.10.1 tqdm==4.48.0

RUN pip install statsmodels==0.10.1 tqdm==4.48.0
RUN pip install requests==2.24.0

# RUN pip install scipy==0.18.1
RUN pip install seaborn==0.9.1
RUN pip install matplotlib==3.1.0

RUN pip install git+https://github.com/macrosynergy/macrosynergy.git@feature/py37 --no-deps

RUN pip install typing-inspect==0.7.1
RUN pip install typing-extensions==4.6.3

RUN pip install scikit-learn==0.20.3 --no-deps

RUN pip install pytest
RUN pip install parameterized
RUN pip install pytest-cov
RUN pip install coverage
RUN pip install pytest-xdist
RUN pip install flake8

# Set the working directory
WORKDIR /app
# Copy the rest of your application code
COPY . /app

# Run the application
# CMD ["python", "test.py"]

CMD ["pytest", "tests"]
