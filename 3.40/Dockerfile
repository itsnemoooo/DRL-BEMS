# Use an official Python base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install dependencies required for EnergyPlus
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Download and extract EnergyPlus 22.1.0
RUN curl -SL https://github.com/NREL/EnergyPlus/archive/refs/tags/v22.1.0-LowFlowTolFix.tar.gz -o eplus.tar.gz \
    && mkdir /usr/local/EnergyPlus-22-1-0 \
    && tar -xzf eplus.tar.gz -C /usr/local/EnergyPlus-22-1-0 --strip-components=1 \
    && rm eplus.tar.gz

# Set the EnergyPlus path environment variable
ENV ENERGYPLUS_ROOT /usr/local/EnergyPlus-22-1-0

# Copy the Python script and any necessary files to the working directory
COPY . /app

# Copy the local pyenergyplus package to the working directory
COPY ./OpenStudio-1.4.0/EnergyPlus/pyenergyplus /app/pyenergyplus

# Install the required Python dependencies (excluding pyenergyplus)
RUN pip install --no-cache-dir \
    numpy \
    pandas \
    matplotlib \
    torch \
    openstudio

# Set the entry point to run your Python script
ENTRYPOINT ["python", "BMW_HVAC_DQN_ONOFF_2.0_HPC.py"]
