# Bird Species Identification System - Project Structure

## Overview

This project implements a multimodal bird species identification system that can recognize bird species from both images and audio recordings. The system uses deep learning models with attention mechanisms to achieve accurate identification.

## Directory Structure

```
bird_identification_system/
├── app.py                  # Main Streamlit application file
├── model.py                # Neural network model implementations
├── utils.py                # Helper functions for data processing and visualization
├── config.py               # Configuration settings and constants
├── README.md               # Project overview and setup instructions
├── requirements.txt        # Python dependencies
├── methodology.md          # Detailed explanation of the technical approach
├── design.md               # System architecture and design principles
├── uml_and_architecture.md # UML diagrams of the system
├── pseudocode.md           # Pseudocode for core algorithms
│
├── models/                 # Saved model weights
│   ├── image_model.pth     # Trained image classification model
│   └── audio_model.pth     # Trained audio classification model
│
├── image/                  # Image data directory
│   ├── attributes.txt      # Image attribute information
│   └── CUB_200_2011/       # Caltech-UCSD Birds-200-2011 dataset structure
│       └── images/         # Bird images organized by species
│           └── 001.Black_footed_Albatross/
│               ├── image1.jpg
│               └── ...
│
├── audio/                  # Audio data directory
│   ├── Birds Voice.csv     # Metadata for bird audio recordings
│   └── Voice of Birds/     # Bird audio recordings
│       ├── sample.wav
│       └── ...
│
└── brid_db/                # Bird species database information
```

## Key Components

### Application Layer

- **app.py**: The Streamlit web interface that allows users to upload images and audio files, displays predictions, and visualizes results. Organized into tabs for identification, bird information, and explanation of the system.

### Model Architecture

- **model.py**: Implements the neural network architectures:
  - `EnhancedResNet`: Image classification model based on ResNet50 with spatial attention
  - `AudioLSTMWithAttention`: Audio classification model using bidirectional LSTM with attention
  - `FusionModule`: Combines predictions from both modalities
  - `BirdSpeciesClassifier`: Main class that orchestrates the models and prediction process

### Data Processing

- **utils.py**: Contains utility functions for:
  - Loading and preprocessing images and audio
  - Creating visualizations of predictions and mel spectrograms
  - Processing model outputs

### Configuration

- **config.py**: Centralizes all configuration settings:
  - File paths and directories
  - Model parameters
  - Training settings
  - Default bird classes

## Data Organization

### Image Data

The image data follows the Caltech-UCSD Birds-200-2011 dataset structure, organized hierarchically:
- Images stored by species in separate directories
- Each directory named with an ID and species name (e.g., "001.Black_footed_Albatross")

### Audio Data

The audio data consists of:
- WAV files in the "Voice of Birds" directory
- A CSV file mapping species to audio files
- Mel spectrograms are generated on-the-fly from the raw audio files

## Documentation

- **methodology.md**: In-depth explanation of the technical approach, algorithms, and model architectures
- **design.md**: Overall system design, principles, and architectural choices
- **uml_and_architecture.md**: UML diagrams (architecture, class, sequence, etc.)
- **pseudocode.md**: Detailed pseudocode for the core algorithms

## System Flow

1. User uploads an image and/or audio file through the web interface
2. The system preprocesses the inputs (resizing images, creating spectrograms from audio)
3. The models generate predictions for each modality
4. If both modalities are available, the fusion module combines the predictions
5. Results are displayed with confidence scores and visualizations
6. Prediction history is stored in the session state

## Extensibility

The system is designed to be extensible:
- New bird species can be added to the configuration
- Models can be retrained with additional data
- The interface can be enhanced with additional features 