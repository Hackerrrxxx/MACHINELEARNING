# Advanced Bird Species Identification System

This application uses state-of-the-art deep learning to identify bird species from both images and audio recordings. It combines an enhanced ResNet50 model with attention mechanisms for image classification and a bidirectional LSTM with attention for audio classification to provide accurate species identification.

## Features

- **Multimodal Recognition**: Process images, audio, or both for highest accuracy
- **Attention Mechanisms**: Helps models focus on the most important features
- **Confidence Visualization**: Visual representation of prediction confidence
- **Prediction History**: Track and review past identifications
- **Species Information**: Access detailed information about identified birds
- **Adaptive Thresholds**: Set minimum confidence thresholds for predictions
- **Visualizations**: View attention maps and spectrograms to understand model decisions

## Technical Highlights

- **Enhanced ResNet50**: Image model with spatial attention
- **Bidirectional LSTM**: Audio model with attention mechanisms
- **Multimodal Fusion**: Intelligent combination of predictions from both models
- **Configurable Settings**: User-adjustable parameters for prediction display
- **Robust Error Handling**: Graceful handling of processing errors
- **Modular Design**: Well-organized codebase with separation of concerns

## Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd bird-species-identification-system
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run app.py
```

2. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:8501)

3. Use the interface to:
   - Upload bird images (JPG, JPEG, PNG)
   - Upload bird audio recordings (WAV, MP3)
   - Adjust prediction settings
   - View prediction history
   - Learn about different bird species

## Project Structure

- `app.py`: Streamlit web application
- `model.py`: Neural network model implementations
- `utils.py`: Utility functions for data processing and visualization
- `config.py`: Configuration parameters
- `requirements.txt`: Project dependencies
- `models/`: Directory for saved model weights

## Models

The system uses three models:
- **Enhanced ResNet50**: For image classification with spatial attention
- **Bidirectional LSTM**: For audio classification with temporal attention
- **Fusion Module**: For combining predictions from multiple modalities

## Requirements

- Python 3.7+
- PyTorch
- Streamlit
- librosa
- matplotlib
- pandas
- Other dependencies listed in requirements.txt

## Future Enhancements

- Geolocation-based filtering for more accurate predictions
- Larger bird species database with more comprehensive information
- Real-time processing using webcam and microphone
- Mobile app integration for field use
- Community features for sharing and discussing identifications

## License

This project is licensed under the MIT License - see the LICENSE file for details. 