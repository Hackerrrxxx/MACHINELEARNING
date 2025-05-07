import torch
import numpy as np
import librosa
import os
from PIL import Image
import matplotlib.pyplot as plt
import io
from torchvision import transforms
import streamlit as st
import config

def load_and_preprocess_image(image_file):
    """Load and preprocess an image for the model."""
    try:
        if image_file is None:
            st.error("No image file provided")
            return None
            
        # Handle different types of image sources
        if hasattr(image_file, 'read'):
            # File is already open (e.g., from streamlit uploader)
            try:
                # Reset file pointer to beginning
                image_file.seek(0)
                image = Image.open(image_file).convert('RGB')
            except Exception as e:
                st.error(f"Error opening uploaded image: {e}")
                return None
        elif isinstance(image_file, str):
            # Path to file or URL
            if image_file.startswith(('http://', 'https://')):
                # Remote URL
                try:
                    import requests
                    from io import BytesIO
                    response = requests.get(image_file, timeout=5)
                    image = Image.open(BytesIO(response.content)).convert('RGB')
                except Exception as e:
                    st.error(f"Error loading image from URL: {e}")
                    return None
            else:
                # Local file path
                try:
                    image = Image.open(image_file).convert('RGB')
                except Exception as e:
                    st.error(f"Error loading image from path: {e}")
                    return None
        elif isinstance(image_file, Image.Image):
            # Already a PIL Image
            image = image_file.convert('RGB')
        elif hasattr(image_file, 'seek') and hasattr(image_file, 'read'):
            # BytesIO or similar
            try:
                image_file.seek(0)
                image = Image.open(image_file).convert('RGB')
            except Exception as e:
                st.error(f"Error loading image from buffer: {e}")
                return None
        else:
            st.error(f"Unsupported image source type: {type(image_file)}")
            return None
            
        # Apply transformations
        try:
            transform = transforms.Compose([
                transforms.Resize(config.IMAGE_SIZE),
                transforms.ToTensor(),
                transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
            ])
            return transform(image).unsqueeze(0)
        except Exception as e:
            st.error(f"Error preprocessing image: {e}")
            return None
    except Exception as e:
        st.error(f"Unexpected error preprocessing image: {e}")
        return None

def load_and_preprocess_audio(audio_file, sample_rate=config.SAMPLE_RATE):
    """Load and preprocess audio for the model."""
    try:
        if hasattr(audio_file, 'read'):
            # Need to save to a temporary file first
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(audio_file.getvalue())
                tmp_filepath = tmp_file.name
            
            y, sr = librosa.load(tmp_filepath, sr=sample_rate)
            os.unlink(tmp_filepath)  # Clean up temp file
        else:
            # Path to file
            y, sr = librosa.load(audio_file, sr=sample_rate)
        
        # Create mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=y, 
            sr=sr, 
            n_mels=config.N_MELS,
            hop_length=config.HOP_LENGTH
        )
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return torch.FloatTensor(mel_spec_db).unsqueeze(0)
    except Exception as e:
        st.error(f"Error preprocessing audio: {e}")
        return None

def plot_mel_spectrogram(audio_tensor):
    """Generate a visualization of the mel spectrogram."""
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(
            audio_tensor.squeeze().numpy(),
            x_axis='time',
            y_axis='mel',
            ax=ax
        )
        fig.colorbar(img, ax=ax, format='%+2.0f dB')
        ax.set(title='Mel-frequency spectrogram')
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf
    except Exception as e:
        st.error(f"Error generating spectrogram visualization: {e}")
        return None

def visualize_image_prediction(image_tensor, pred_class, confidence):
    """Generate a visualization for image prediction explanation."""
    try:
        # Convert tensor to numpy for visualization
        img_np = image_tensor.squeeze().permute(1, 2, 0).numpy()
        # Denormalize
        img_np = np.clip(img_np * np.array(config.IMAGE_STD) + np.array(config.IMAGE_MEAN), 0, 1)
        
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(img_np)
        ax.set_title(f"Prediction: {pred_class}\nConfidence: {confidence:.2f}%")
        ax.axis('off')
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf
    except Exception as e:
        st.error(f"Error generating visualization: {e}")
        # Return a fallback visualization with just text
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.text(0.5, 0.5, f"Prediction: {pred_class}\nConfidence: {confidence:.2f}%", 
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return buf

def get_top_predictions(prediction_tensor, class_names, top_k=3):
    """Get the top k predictions from the model output."""
    try:
        probabilities, indices = torch.topk(prediction_tensor, top_k)
        results = []
        for i in range(top_k):
            idx = indices[0][i].item()
            prob = probabilities[0][i].item() * 100
            results.append((class_names[idx], prob))
        return results
    except Exception as e:
        st.error(f"Error getting top predictions: {e}")
        return [(class_names[0], 0.0)] 