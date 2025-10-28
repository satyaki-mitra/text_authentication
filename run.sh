#!/bin/bash

echo "Starting Text Auth AI Detection System..."

# Check if Conda is installed
if ! command -v conda &> /dev/null; then
    echo "Conda is required but not installed. Please install Miniconda or Anaconda."
    exit 1
fi

# Check if Python is installed and is version 3.10+
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not installed. Please install Python 3.10 or higher."
    exit 1
fi
python3 -c "import sys; assert sys.version_info >= (3.10,), 'Python 3.10 or higher is required.'" || exit 1

# Conda environment name
CONDA_ENV_NAME="text_auth_env"

# Check if conda environment exists, create if not
if ! conda info --envs | grep -q "$CONDA_ENV_NAME"; then
    echo "Creating Conda environment '$CONDA_ENV_NAME' with Python 3.10..."
    conda create -n "$CONDA_ENV_NAME" python=3.10 -y
fi

# Activate conda environment
echo "Activating Conda environment '$CONDA_ENV_NAME'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$CONDA_ENV_NAME"

# Install requirements
echo "Installing dependencies..."
pip install -r requirements.txt || { echo "Failed to install dependencies."; exit 1; }

# Create necessary directories
mkdir -p logs
mkdir -p data/uploads
mkdir -p data/reports
mkdir -p models/cache

# Set environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export MODEL_CACHE_DIR=$(pwd)/models/cache

# Start the FastAPI application
echo "Starting FastAPI server..."
echo "Access the application at: http://localhost:8000"
echo "API documentation at: http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"

# Deactivate conda environment on exit
trap 'conda deactivate' EXIT

uvicorn app:app --reload --host 0.0.0.0 --port 8000