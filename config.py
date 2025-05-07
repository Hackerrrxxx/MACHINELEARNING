import os

# Paths
IMAGE_DIR = 'image'
AUDIO_DIR = 'audio'
CSV_FILE = os.path.join(AUDIO_DIR, 'Birds Voice.csv')
MODEL_DIR = 'models'

# Ensure model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Model paths
IMAGE_MODEL_PATH = os.path.join(MODEL_DIR, 'image_model.pth')
AUDIO_MODEL_PATH = os.path.join(MODEL_DIR, 'audio_model.pth')

# Model settings
NUM_CLASSES = 5
DEFAULT_CLASSES = {
    0: '001.Black_footed_Albatross',
    1: '002.Laysan_Albatross',
    2: '003.Sooty_Albatross',
    3: '004.Groove_billed_Ani',
    4: '005.Crested_Auklet'
}

# Image settings
IMAGE_SIZE = (224, 224)
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]

# Audio settings
SAMPLE_RATE = 22050  # Hz
N_MELS = 128  # Number of mel bands
HOP_LENGTH = 512  # Hop length for STFT

# Training settings
BATCH_SIZE = 32
NUM_WORKERS = 4
NUM_EPOCHS = 10
LEARNING_RATE = 0.001 