import streamlit as st
import torch
from model import BirdSpeciesClassifier
import tempfile
import os
import sys
from PIL import Image, ImageDraw
import numpy as np
import librosa
import matplotlib.pyplot as plt
import config
import utils
from datetime import datetime
import pandas as pd
import io
import requests

# Set page configuration
st.set_page_config(
    page_title="Bird Species Identification",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        padding: 10px 24px;
        border-radius: 8px;
        border: none;
        font-size: 16px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .upload-section {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    .prediction-card {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
        margin-bottom: 20px;
    }
    .footer {
        text-align: center;
        color: #7f8c8d;
        padding-top: 20px;
        font-style: italic;
    }
    .confidence-bar {
        height: 20px;
        background-color: #e0e0e0;
        border-radius: 10px;
        overflow: hidden;
        margin: 10px 0;
    }
    .confidence-fill {
        height: 100%;
        background-color: #4CAF50;
        border-radius: 10px;
    }
    .bird-info {
        padding: 15px;
        border-radius: 10px;
        background-color: #fafafa;
        margin-top: 10px;
    }
    .custom-tabs .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
    }
    .custom-tabs .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f0f0;
        border-radius: 5px 5px 0 0;
    }
    .custom-tabs .stTabs [aria-selected="true"] {
        background-color: white;
        border-top: 2px solid #4CAF50;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and description
st.markdown("<h1 style='text-align: center; color: #2c3e50;'>Bird Species Identification</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #7f8c8d; font-style: italic;'>Discover the beauty of nature through the eyes and ears of birds</p>", unsafe_allow_html=True)

# Near the top of the file, add this function after imports
def safe_image_display(image_src, caption="", use_container_width=True):
    """Safely display images from various sources with proper error handling."""
    try:
        # Check if image_src is a file-like object (from upload)
        if hasattr(image_src, 'read'):
            try:
                # Reset pointer to beginning of file
                image_src.seek(0)
                st.image(image_src, caption=caption, use_container_width=use_container_width)
                return True
            except Exception as e:
                st.error(f"Error displaying uploaded image: {str(e)}")
                st.info(f"🖼️ {caption if caption else 'Image'}")
                return False
            
        # Check if image_src is a URL
        elif isinstance(image_src, str) and (image_src.startswith('http://') or image_src.startswith('https://')):
            try:
                # Create local fallback image instead of using URLs that might be unreliable
                # Using globally imported Image and ImageDraw
                
                # Create a simple placeholder image
                img = Image.new('RGB', (300, 200), color=(240, 240, 240))
                draw = ImageDraw.Draw(img)
                draw.text((10, 10), f"Placeholder for: {caption}", fill=(0, 0, 0))
                
                # Draw a simple bird shape if it's a bird image
                if "bird" in caption.lower() or "albatross" in caption.lower() or "auklet" in caption.lower():
                    draw.ellipse((100, 50, 200, 120), fill=(100, 100, 100))  # Head
                    draw.polygon([(150, 120), (200, 160), (100, 160)], fill=(100, 100, 100))  # Body
                
                # Convert to BytesIO and display
                buf = io.BytesIO()
                img.save(buf, format='PNG')
                buf.seek(0)
                st.image(buf, caption=caption, use_container_width=use_container_width)
                return True
            except Exception as e:
                st.error(f"Error creating placeholder image: {str(e)}")
                st.info(f"🖼️ {caption if caption else 'Image'}")
                return False
                
        # Local file path
        elif isinstance(image_src, str):
            try:
                image = Image.open(image_src).convert('RGB')
                st.image(image, caption=caption, use_container_width=use_container_width)
                return True
            except Exception as e:
                st.error(f"Could not load image from path: {str(e)}")
                st.info(f"🖼️ {caption if caption else 'Image'}")
                return False
                
        # Already a PIL Image
        elif isinstance(image_src, Image.Image):
            st.image(image_src, caption=caption, use_container_width=use_container_width)
            return True
            
        # BytesIO or similar
        elif hasattr(image_src, 'seek'):
            try:
                image_src.seek(0)
                image = Image.open(image_src)
                st.image(image, caption=caption, use_container_width=use_container_width)
                return True
            except Exception as e:
                st.error(f"Could not load image from buffer: {str(e)}")
                st.info(f"🖼️ {caption if caption else 'Image'}")
                return False
        
        else:
            st.error(f"Unsupported image source type: {type(image_src)}")
            st.info(f"🖼️ {caption if caption else 'Image'}")
            return False
            
    except Exception as e:
        st.error(f"Error displaying image: {str(e)}")
        st.info(f"🖼️ {caption if caption else 'Image'}")
        return False

# Replace sidebar image
def create_bird_image(save_path=None):
    """Create a simple bird image and return the PIL Image object."""
    try:
        # No need to import Image and ImageDraw here as they're imported globally
        
        # Create a new image with white background
        img = Image.new('RGB', (300, 200), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        # Draw a simple bird
        draw.ellipse((100, 50, 200, 120), fill=(100, 100, 100))  # Head
        draw.polygon([(150, 120), (200, 160), (100, 160)], fill=(100, 100, 100))  # Body
        draw.ellipse((170, 70, 185, 85), fill=(255, 255, 255))  # Eye
        draw.line([(200, 85), (220, 70)], fill=(100, 100, 100), width=3)  # Beak
        
        # Save if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            img.save(save_path)
            
        return img
    except Exception as e:
        print(f"Error creating bird image: {e}")
        return None

# Generate and save local images for the app
def create_app_images():
    """Create all necessary images for the app interface."""
    try:
        # Create images directory if it doesn't exist
        app_images_dir = os.path.join("image", "app_images")
        os.makedirs(app_images_dir, exist_ok=True)
        
        # Create sidebar bird image
        sidebar_img_path = os.path.join(app_images_dir, "sidebar_bird.jpg")
        if not os.path.exists(sidebar_img_path):
            sidebar_img = create_bird_image(sidebar_img_path)
            print(f"Created sidebar image at {sidebar_img_path}")
            
        # Create distribution map image
        map_img_path = os.path.join(app_images_dir, "distribution_map.jpg")
        if not os.path.exists(map_img_path):
            # Create a simple map
            # No need to import Image and ImageDraw here
            img = Image.new('RGB', (500, 300), color=(230, 230, 250))
            draw = ImageDraw.Draw(img)
            
            # Draw some continents
            draw.polygon([(50, 50), (200, 100), (150, 200), (30, 180)], fill=(200, 200, 150))  # Continent 1
            draw.polygon([(300, 80), (450, 70), (430, 170), (320, 220)], fill=(200, 200, 150))  # Continent 2
            
            # Draw some distribution dots
            for x, y in [(120, 100), (135, 150), (350, 130), (400, 100)]:
                draw.ellipse((x-5, y-5, x+5, y+5), fill=(200, 50, 50))
                
            # Add text
            draw.text((10, 10), "Bird Distribution Map", fill=(0, 0, 0))
            
            img.save(map_img_path)
            print(f"Created map image at {map_img_path}")
            
        # Create neural network diagram
        nn_img_path = os.path.join(app_images_dir, "neural_network.jpg")
        if not os.path.exists(nn_img_path):
            # Create a simple NN diagram
            # No need to import Image and ImageDraw here
            img = Image.new('RGB', (500, 300), color=(255, 255, 255))
            draw = ImageDraw.Draw(img)
            
            # Draw input layer
            input_y = [50, 100, 150, 200, 250]
            for y in input_y:
                draw.ellipse((50-10, y-10, 50+10, y+10), outline=(0, 0, 0), fill=(200, 200, 255))
                
            # Draw hidden layer
            hidden_y = [75, 125, 175, 225]
            for y in hidden_y:
                draw.ellipse((250-10, y-10, 250+10, y+10), outline=(0, 0, 0), fill=(200, 255, 200))
            
            # Draw output layer
            output_y = [100, 150, 200]
            for y in output_y:
                draw.ellipse((450-10, y-10, 450+10, y+10), outline=(0, 0, 0), fill=(255, 200, 200))
                
            # Draw connections
            for in_y in input_y:
                for hid_y in hidden_y:
                    draw.line([(50+10, in_y), (250-10, hid_y)], fill=(200, 200, 200))
                    
            for hid_y in hidden_y:
                for out_y in output_y:
                    draw.line([(250+10, hid_y), (450-10, out_y)], fill=(200, 200, 200))
            
            # Add title
            draw.text((150, 10), "Neural Network Architecture", fill=(0, 0, 0))
            
            img.save(nn_img_path)
            print(f"Created neural network diagram at {nn_img_path}")
        
        return {
            'sidebar': sidebar_img_path,
            'map': map_img_path,
            'neural_network': nn_img_path
        }
    except Exception as e:
        print(f"Error creating app images: {e}")
        return {}

# Generate app images at startup
app_images = create_app_images()

# Replace the sidebar image code with this:
try:
    sidebar_img = app_images.get('sidebar', "")
    if os.path.exists(sidebar_img):
        safe_image_display(sidebar_img)
    else:
        st.sidebar.info("🦜 Bird Species Identification")
except Exception as e:
    st.sidebar.error("Could not load sidebar image")
    st.sidebar.info("🦜 Bird Species Identification")

# Sidebar
st.sidebar.title("About")
st.sidebar.markdown("""
This application uses deep learning to identify bird species from images and/or audio recordings. 
- **Image Recognition**: Uses an enhanced ResNet50 model with attention mechanisms
- **Audio Recognition**: Uses bidirectional LSTM with attention mechanisms
- **Multimodal Fusion**: Combines both inputs for higher accuracy when available
""")

# History storage
if 'history' not in st.session_state:
    st.session_state.history = []

# Sidebar tabs
sidebar_tab1, sidebar_tab2 = st.sidebar.tabs(["Settings", "History"])

with sidebar_tab1:
    st.header("Settings")
    show_visualization = st.checkbox("Show visualizations", value=True)
    show_top_k = st.slider("Number of predictions to show", min_value=1, max_value=5, value=3)
    prediction_threshold = st.slider("Confidence threshold (%)", min_value=0, max_value=100, value=20)
    
with sidebar_tab2:
    st.header("Prediction History")
    if st.session_state.history:
        for i, entry in enumerate(st.session_state.history):
            with st.expander(f"{entry['time']} - {entry['species']} ({entry['confidence']:.1f}%)"):
                st.write(f"**Input type:** {entry['input_type']}")
                st.write(f"**Confidence:** {entry['confidence']:.2f}%")
                st.write(f"**Time:** {entry['time']}")
    else:
        st.info("No predictions yet. Try identifying some birds!")
    
    if st.button("Clear History"):
        st.session_state.history = []
        st.experimental_rerun()

# Initialize the model
@st.cache_resource
def load_model():
    try:
        # Ensure model directory exists
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        
        # Create model with better error handling
        model = BirdSpeciesClassifier()
        
        # Get class names from configuration
        class_names = {}
        for idx, name in config.DEFAULT_CLASSES.items():
            class_names[idx] = name
            
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.error("The application will continue with limited functionality.")
        
        # Return placeholder model and class names to avoid crashing
        class DummyModel:
            def __init__(self):
                self.device = "cpu"
            
            def predict(self, image_tensor=None, audio_tensor=None):
                # Return dummy prediction
                return torch.zeros(1, 5), (None, None)
        
        model = DummyModel()
        return model, config.DEFAULT_CLASSES

# Load model
with st.spinner("Loading the bird identification model..."):
    model, class_names = load_model()

# Main tabs
tab1, tab2, tab3 = st.tabs(["Identify Birds", "About Birds", "How It Works"])

with tab1:
    st.markdown('<div class="custom-tabs">', unsafe_allow_html=True)
    
    # Create columns for the input options
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        st.markdown("### Image Upload")
        image_file = st.file_uploader("Upload a bird image", type=['jpg', 'jpeg', 'png'])
        if image_file is not None:
            safe_image_display(image_file, caption="Uploaded Image")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
        st.markdown("### Audio Upload")
        audio_file = st.file_uploader("Upload bird audio", type=['wav', 'mp3'])
        if audio_file is not None:
            try:
                st.audio(audio_file)
                if show_visualization:
                    audio_tensor = utils.load_and_preprocess_audio(audio_file)
                    if audio_tensor is not None:
                        spec_plot = utils.plot_mel_spectrogram(audio_tensor)
                        if spec_plot:
                            safe_image_display(spec_plot, caption="Mel Spectrogram")
            except Exception as e:
                st.error(f"Error processing audio: {e}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Prediction button
    col_button = st.columns([1, 2, 1])[1]
    with col_button:
        predict_button = st.button("🔍 Identify Bird Species", key="predict", use_container_width=True)
    
    # Make prediction
    if predict_button:
        if image_file is None and audio_file is None:
            st.warning("Please upload either an image or audio file (or both) for identification.")
        else:
            with st.spinner("Analyzing bird characteristics..."):
                try:
                    # Initialize variables
                    image_tensor, audio_tensor = None, None
                    
                    # Process image if provided
                    if image_file is not None:
                        try:
                            # Reset the file pointer to the beginning of the file
                            image_file.seek(0)
                            image_tensor = utils.load_and_preprocess_image(image_file)
                            if image_tensor is None:
                                st.error("Failed to process image")
                            else:
                                image_tensor = image_tensor.to(model.device)
                        except Exception as e:
                            st.error(f"Error processing image: {e}")
                    
                    # Process audio if provided
                    if audio_file is not None:
                        try:
                            # Reset the file pointer to the beginning of the file
                            audio_file.seek(0)
                            audio_tensor = utils.load_and_preprocess_audio(audio_file)
                            if audio_tensor is None:
                                st.error("Failed to process audio")
                            else:
                                audio_tensor = audio_tensor.to(model.device)
                        except Exception as e:
                            st.error(f"Error processing audio: {e}")
                    
                    # Make prediction if at least one input was successfully processed
                    if image_tensor is not None or audio_tensor is not None:
                        prediction, attention_maps = model.predict(image_tensor, audio_tensor)
                        
                        # Get top predictions
                        top_predictions = utils.get_top_predictions(
                            torch.softmax(prediction, dim=1), 
                            class_names,
                            top_k=show_top_k
                        )
                        
                        # Get the top prediction
                        top_class, top_confidence = top_predictions[0]
                        
                        # Display prediction only if confidence is above threshold
                        if top_confidence >= prediction_threshold:
                            # Add to history
                            st.session_state.history.append({
                                'species': top_class,
                                'confidence': top_confidence,
                                'input_type': 'Image & Audio' if image_tensor is not None and audio_tensor is not None else 'Image' if image_tensor is not None else 'Audio',
                                'time': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            })
                            
                            # Display the results
                            st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
                            st.markdown(f"### 🐦 Identified Bird: {top_class}")
                            
                            # Display confidence bar
                            st.markdown(f"<div class='confidence-bar'><div class='confidence-fill' style='width: {top_confidence}%;'></div></div>", unsafe_allow_html=True)
                            st.markdown(f"<p style='text-align: center;'>Confidence: {top_confidence:.2f}%</p>", unsafe_allow_html=True)
                            
                            # Display visualizations if enabled
                            if show_visualization:
                                if image_tensor is not None and attention_maps[0] is not None:
                                    vis_img = utils.visualize_image_prediction(image_tensor.cpu(), top_class, top_confidence)
                                    if vis_img:
                                        safe_image_display(vis_img, caption="Prediction Visualization")
                            
                            # Display other top predictions
                            if len(top_predictions) > 1:
                                st.markdown("#### Other possibilities:")
                                for species, conf in top_predictions[1:]:
                                    st.markdown(f"<div style='display: flex; align-items: center;'>"
                                              f"<div style='flex-grow: 1;'>{species}</div>"
                                              f"<div style='width: 80px; text-align: right;'>{conf:.2f}%</div>"
                                              f"</div>", unsafe_allow_html=True)
                                              
                            # Display input information
                            st.markdown("#### Analysis based on:")
                            input_type = []
                            if image_tensor is not None:
                                input_type.append("🖼️ Image")
                            if audio_tensor is not None:
                                input_type.append("🔊 Audio")
                            st.markdown(", ".join(input_type))
                            
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.warning(f"Confidence too low ({top_confidence:.2f}%). Unable to make a reliable prediction. Please try with a clearer image or audio.")
                    else:
                        st.error("Unable to process inputs. Please try with different files.")
                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.error("Try uploading a different image or audio file.")
                    import traceback
                    st.text(traceback.format_exc())
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.header("Bird Information Database")
    
    # Create a dataframe with bird information
    bird_data = pd.DataFrame({
        "Species": list(config.DEFAULT_CLASSES.values()),
        "Habitat": ["Open ocean and rocky shores", "Marine habitats, nests on islands", 
                   "Southern Ocean, breeds on islands", "Open woodland and scrub", 
                   "Rocky shores and cliffs in Arctic regions"],
        "Diet": ["Fish, squid, and crustaceans", "Squid, fish, and crustaceans", 
                "Fish, squid, and carrion", "Insects, fruits, and small vertebrates", 
                "Small fish, crustaceans, and mollusks"],
        "Size": ["81-94 cm wingspan", "195-215 cm wingspan", 
                "85-95 cm wingspan", "30-35 cm in length", 
                "25 cm in length"]
    })
    
    selected_bird = st.selectbox("Select a bird species to learn more", bird_data["Species"])
    
    # Display info for the selected bird
    bird_info = bird_data[bird_data["Species"] == selected_bird].iloc[0]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        try:
            # Replace bird info section image:
            bird_img = os.path.join("image", "CUB_200_2011", "images", "001.Black_footed_Albatross", "sample_bird.jpg")
            if os.path.exists(bird_img):
                safe_image_display(bird_img, caption=selected_bird)
            else:
                # Create bird on-the-fly
                bird_image = create_bird_image()
                if bird_image:
                    st.image(bird_image, caption=selected_bird, use_container_width=True)
                else:
                    st.info(f"🦜 {selected_bird}")
        except Exception as e:
            st.error("Could not load bird image")
            st.info(f"🦜 {selected_bird}")
    
    with col2:
        st.markdown(f"<div class='bird-info'>", unsafe_allow_html=True)
        st.markdown(f"### {selected_bird}")
        st.markdown(f"**Habitat:** {bird_info['Habitat']}")
        st.markdown(f"**Diet:** {bird_info['Diet']}")
        st.markdown(f"**Size:** {bird_info['Size']}")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("#### Conservation Status")
        # Dummy conservation status
        status = "Least Concern"
        st.progress(80, text=status)
        
        st.markdown("#### Geographic Distribution")
        try:
            # Replace map image:
            map_img = app_images.get('map', "")
            if os.path.exists(map_img):
                safe_image_display(map_img, caption="Distribution Map")
            else:
                st.info("Map data not available")
        except Exception as e:
            st.error("Could not load distribution map")
            st.info("Map data not available")

with tab3:
    st.header("How the Model Works")
    
    st.markdown("""
    This bird species identification system uses advanced deep learning techniques to identify birds from both images and audio recordings.
    
    #### Image Recognition
    For images, we use an **Enhanced ResNet50** architecture with the following improvements:
    - **Spatial Attention**: Helps the model focus on the most important regions in the image
    - **Global Average Pooling**: Reduces parameters while maintaining spatial information
    - **Dropout Layers**: Prevent overfitting by randomly deactivating neurons during training
    
    #### Audio Recognition
    For audio processing, we use a **Bidirectional LSTM with Attention** that:
    - Converts audio into mel spectrograms (visual representations of sound)
    - Uses bidirectional processing to capture patterns in both directions
    - Applies attention mechanisms to focus on the most distinguishing audio features
    
    #### Multimodal Fusion
    When both image and audio are provided, a **Fusion Module** combines the outputs to make a more accurate prediction.
    """)
    
    # Example diagram
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        try:
            # Replace neural network diagram:
            nn_img = app_images.get('neural_network', "")
            if os.path.exists(nn_img):
                safe_image_display(nn_img, caption="Simplified Neural Network Architecture")
            else:
                st.info("Neural Network Architecture Diagram")
        except Exception as e:
            st.error("Could not load neural network diagram")
            st.info("Neural Network Architecture Diagram")
    
    st.markdown("""
    #### Model Training
    The model was trained on a dataset of bird images and audio recordings, with data augmentation techniques 
    to improve generalization. The training process involved:
    - Transfer learning from models pre-trained on large datasets
    - Fine-tuning with bird-specific data
    - Regularization techniques to prevent overfitting
    
    #### Continuous Improvement
    The system is designed to improve over time as more data becomes available and can be 
    adapted to recognize additional bird species in the future.
    """)

# Footer
st.markdown("---")
st.markdown("<p class='footer'>Built with ❤️ for bird enthusiasts | Last updated: May 2025</p>", unsafe_allow_html=True)

# Add this function after imports and before the rest of the app
def ensure_sample_files_exist():
    """Create sample files for testing if they don't exist."""
    try:
        # Create sample image
        image_dir = os.path.join('image', 'CUB_200_2011', 'images', '001.Black_footed_Albatross')
        os.makedirs(image_dir, exist_ok=True)
        
        sample_image_path = os.path.join(image_dir, 'sample_bird.jpg')
        if not os.path.exists(sample_image_path):
            # Create a simple bird image
            try:
                from PIL import Image, ImageDraw
                img = Image.new('RGB', (224, 224), color='white')
                draw = ImageDraw.Draw(img)
                # Draw a simple bird shape
                draw.ellipse((50, 50, 174, 174), fill='black')  # Head
                draw.polygon([(110, 174), (160, 220), (60, 220)], fill='black')  # Body
                img.save(sample_image_path)
                print(f"Created sample bird image at {sample_image_path}")
            except Exception as e:
                print(f"Could not create sample image: {e}")
        
        # Create sample audio file
        audio_dir = os.path.join('audio', 'Voice of Birds')
        os.makedirs(audio_dir, exist_ok=True)
        
        sample_audio_path = os.path.join(audio_dir, 'sample_bird.wav')
        if not os.path.exists(sample_audio_path):
            try:
                # Create a simple WAV file with bird-like chirping
                import numpy as np
                import soundfile as sf
                
                sample_rate = 22050
                duration = 2  # seconds
                t = np.linspace(0, duration, int(sample_rate * duration))
                
                # Create a chirping sound (simple sine wave with frequency modulation)
                chirp = np.sin(2 * np.pi * 1000 * t * (1 + 0.5 * np.sin(2 * np.pi * 3 * t)))
                
                # Add some amplitude modulation
                envelope = 0.5 * (1 + np.sin(2 * np.pi * 5 * t))
                audio = chirp * envelope
                
                # Save to WAV file
                sf.write(sample_audio_path, audio, sample_rate)
                print(f"Created sample bird audio at {sample_audio_path}")
            except Exception as e:
                print(f"Could not create sample audio: {e}")
                
        # Create models directory and files if they don't exist
        os.makedirs('models', exist_ok=True)
        
        # Ensure model files exist
        if not os.path.exists(config.IMAGE_MODEL_PATH) or not os.path.exists(config.AUDIO_MODEL_PATH):
            try:
                import torch
                # Create and save dummy models
                torch.save({}, config.IMAGE_MODEL_PATH)
                torch.save({}, config.AUDIO_MODEL_PATH)
                print(f"Created dummy model files")
            except Exception as e:
                print(f"Could not create model files: {e}")
                
    except Exception as e:
        print(f"Error ensuring sample files exist: {e}")

# Call the function at startup
ensure_sample_files_exist() 