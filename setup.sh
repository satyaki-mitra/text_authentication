#!/bin/bash

# Post-installation setup script for Hugging Face Spaces 
echo "Starting setup for Text-Authentication Platform ..."

# Download Spacy Model
echo "Downloading SpaCy English model ..."
python -n spacy download en_core_web_sm

# Download NLTK data
echo "Downloading NLTK data ..."
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('averaged_perceptron_tagger')"

# Create necessary directories
echo "Creating directories ..."
mkdir -p data/reports data/uploads

# Verify installation
echo "Verifying installations ..."
python -c "import transformers; import torch; import spacy; print('All core libraries imported successfully.')"

echo "Setup complete !"
