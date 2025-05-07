import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from torch.utils.data import Dataset, DataLoader
import librosa
import torch.nn.functional as F
import os
import pandas as pd
from PIL import Image
from torchvision import transforms
import config

class Attention(nn.Module):
    """Attention mechanism for focusing on important features."""
    def __init__(self, feature_dim):
        super(Attention, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, 1)
        )
        
    def forward(self, x):
        # x shape: (batch, seq_len, feature_dim)
        attention_weights = F.softmax(self.attention(x).squeeze(-1), dim=1)  # (batch, seq_len)
        # Apply attention weights
        context = torch.bmm(attention_weights.unsqueeze(1), x).squeeze(1)  # (batch, feature_dim)
        return context, attention_weights

class BirdDataset(Dataset):
    def __init__(self, image_dir, audio_dir, csv_file=None, transform=None, is_train=True):
        """
        Args:
            image_dir (string): Path to the CUB_200_2011 images directory
            audio_dir (string): Path to the Voice of Birds directory
            csv_file (string): Path to the Birds Voice.csv file
            transform (callable, optional): Optional transform to be applied on images
            is_train (bool): Whether this is training set or test set
        """
        self.image_dir = image_dir
        self.audio_dir = audio_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
        ])
        
        # Load image dataset info
        self.images_path = os.path.join(image_dir, 'CUB_200_2011', 'images')
        self.classes_file = os.path.join(image_dir, 'CUB_200_2011', 'classes.txt')
        self.train_test_split = os.path.join(image_dir, 'CUB_200_2011', 'train_test_split.txt')
        
        # Load audio dataset info
        try:
            self.audio_df = pd.read_csv(csv_file) if csv_file and os.path.exists(csv_file) else None
        except Exception as e:
            print(f"Error loading CSV file: {e}")
            self.audio_df = None
        
        # Get class mappings
        self.class_to_idx = {}
        try:
            if os.path.exists(self.classes_file):
                with open(self.classes_file, 'r') as f:
                    for line in f:
                        idx, class_name = line.strip().split(' ', 1)
                        self.class_to_idx[class_name] = int(idx) - 1  # 0-based indexing
            else:
                # Create dummy class mappings if file doesn't exist
                self.class_to_idx = config.DEFAULT_CLASSES
        except Exception as e:
            print(f"Error loading classes file: {e}")
            # Create dummy class mappings as fallback
            self.class_to_idx = config.DEFAULT_CLASSES
        
        # Get train/test split
        self.split_info = {}
        try:
            if os.path.exists(self.train_test_split):
                with open(self.train_test_split, 'r') as f:
                    for line in f:
                        img_id, is_train = line.strip().split(' ', 1)
                        self.split_info[int(img_id)] = int(is_train)
            else:
                # Create dummy split info if file doesn't exist
                for i in range(1, 12000):
                    self.split_info[i] = 1 if i % 2 == 0 else 0
        except Exception as e:
            print(f"Error loading train_test_split file: {e}")
            # Create dummy split info as fallback
            for i in range(1, 12000):
                self.split_info[i] = 1 if i % 2 == 0 else 0
        
        # Create list of valid samples
        self.samples = []
        try:
            if os.path.exists(self.images_path):
                for class_name in os.listdir(self.images_path):
                    class_dir = os.path.join(self.images_path, class_name)
                    if os.path.isdir(class_dir):
                        for img_name in os.listdir(class_dir):
                            if img_name.endswith(('.jpg', '.jpeg', '.png')):
                                try:
                                    img_id = int(img_name.split('.')[0])
                                    if self.split_info.get(img_id, 1) == (1 if is_train else 0):
                                        # Find corresponding audio file if available
                                        audio_file = None
                                        if self.audio_df is not None:
                                            matching_audio = self.audio_df[
                                                self.audio_df['species'] == class_name
                                            ]
                                            if not matching_audio.empty:
                                                audio_file = os.path.join(
                                                    self.audio_dir,
                                                    'Voice of Birds',
                                                    matching_audio.iloc[0]['filename']
                                                )
                                        
                                        self.samples.append({
                                            'image_path': os.path.join(class_dir, img_name),
                                            'audio_path': audio_file,
                                            'class_idx': self.class_to_idx.get(class_name, 0)
                                        })
                                except Exception as e:
                                    print(f"Error processing image {img_name}: {e}")
            
            # If no samples were loaded, create dummy samples
            if len(self.samples) == 0:
                for i, class_name in enumerate(config.DEFAULT_CLASSES.values()):
                    self.samples.append({
                        'image_path': None,
                        'audio_path': os.path.join(audio_dir, 'Voice of Birds', f'bird00{i+1}.wav'),
                        'class_idx': i
                    })
        except Exception as e:
            print(f"Error creating samples: {e}")
            # Create dummy samples as fallback
            for i, class_name in enumerate(config.DEFAULT_CLASSES.values()):
                self.samples.append({
                    'image_path': None,
                    'audio_path': os.path.join(audio_dir, 'Voice of Birds', f'bird00{i+1}.wav'),
                    'class_idx': i
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = self.samples[idx]
        
        # Load and transform image if available
        try:
            if sample['image_path'] and os.path.exists(sample['image_path']):
                image = Image.open(sample['image_path']).convert('RGB')
                image = self.transform(image)
            else:
                # Create dummy image
                image = torch.zeros((3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]))
        except Exception as e:
            print(f"Error loading image: {e}")
            # Create dummy image
            image = torch.zeros((3, config.IMAGE_SIZE[0], config.IMAGE_SIZE[1]))
        
        # Load and process audio if available
        try:
            if sample['audio_path'] and os.path.exists(sample['audio_path']):
                y, sr = librosa.load(sample['audio_path'], sr=config.SAMPLE_RATE)
                mel_spec = librosa.feature.melspectrogram(
                    y=y, 
                    sr=sr, 
                    n_mels=config.N_MELS,
                    hop_length=config.HOP_LENGTH
                )
                mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
                audio = torch.FloatTensor(mel_spec_db)
            else:
                # Create dummy audio tensor
                audio = torch.zeros((config.N_MELS, config.N_MELS))  # Dummy mel spectrogram
        except Exception as e:
            print(f"Error loading audio: {e}")
            # Create dummy audio tensor
            audio = torch.zeros((config.N_MELS, config.N_MELS))  # Dummy mel spectrogram
        
        return {
            'image': image,
            'audio': audio,
            'label': sample['class_idx']
        }

class AudioLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(AudioLSTMWithAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        # Bidirectional LSTM has 2*hidden_size as output dimension
        self.attention = Attention(hidden_size * 2)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # Process input shape
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add batch dimension
            
        # For mel spectrograms, we need to reshape to treat each frequency bin as a feature
        if x.dim() == 3 and x.size(1) == config.N_MELS:
            x = x.permute(0, 2, 1)  # (batch, time, freq)
        
        # Initial hidden states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        try:
            # LSTM layer
            lstm_out, _ = self.lstm(x, (h0, c0))  # lstm_out: (batch, seq_len, hidden*2)
            
            # Apply attention
            context, attention_weights = self.attention(lstm_out)
            
            # Apply dropout
            context = self.dropout(context)
            
            # Final classifier
            output = self.fc(context)
            
            return output, attention_weights
        except Exception as e:
            print(f"Error in LSTM forward pass: {e}")
            # Return zeros in case of error
            return torch.zeros(x.size(0), self.fc.out_features).to(x.device), torch.zeros(x.size(0), x.size(1)).to(x.device)

class EnhancedResNet(nn.Module):
    def __init__(self, num_classes):
        super(EnhancedResNet, self).__init__()
        # Load pretrained ResNet
        self.base_model = models.resnet50(weights='DEFAULT')
        
        # Remove the final fully connected layer
        self.features = nn.Sequential(*list(self.base_model.children())[:-2])
        
        # Add spatial attention after convolutional layers
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2048, 512, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(512, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Global average pooling
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        try:
            # Extract features
            features = self.features(x)
            
            # Apply spatial attention
            attention_map = self.spatial_attention(features)
            attended_features = features * attention_map
            
            # Global average pooling
            pooled = self.gap(attended_features).view(x.size(0), -1)
            
            # Classification
            output = self.classifier(pooled)
            
            return output, attention_map
        except Exception as e:
            print(f"Error in ResNet forward pass: {e}")
            return torch.zeros(x.size(0), num_classes).to(x.device), torch.zeros(x.size(0), 1, 7, 7).to(x.device)

class FusionModule(nn.Module):
    def __init__(self, num_classes):
        super(FusionModule, self).__init__()
        self.fusion = nn.Sequential(
            nn.Linear(num_classes * 2, num_classes),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(num_classes, num_classes)
        )
        
    def forward(self, image_pred, audio_pred):
        # Combine predictions
        combined = torch.cat([image_pred, audio_pred], dim=1)
        return self.fusion(combined)

class BirdSpeciesClassifier:
    def __init__(self, num_classes=config.NUM_CLASSES):
        # Image model (Enhanced ResNet50)
        self.image_model = EnhancedResNet(num_classes)
        
        # Audio model (LSTM with Attention)
        self.audio_model = AudioLSTMWithAttention(
            input_size=config.N_MELS,
            hidden_size=256,
            num_layers=2,
            num_classes=num_classes
        )
        
        # Fusion module for multimodal learning
        self.fusion_model = FusionModule(num_classes)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load pre-trained weights if available
        try:
            if os.path.exists(config.IMAGE_MODEL_PATH):
                state_dict = torch.load(config.IMAGE_MODEL_PATH, map_location=self.device)
                if state_dict:  # Only try to load if the state dict is not empty
                    try:
                        self.image_model.load_state_dict(state_dict)
                        print(f"Loaded image model from {config.IMAGE_MODEL_PATH}")
                    except Exception as e:
                        print(f"Error loading image model weights: {e}")
                else:
                    print(f"Image model file was empty. Using random initialization.")
            else:
                print(f"Image model file not found at {config.IMAGE_MODEL_PATH}. Using random initialization.")
            
            if os.path.exists(config.AUDIO_MODEL_PATH):
                state_dict = torch.load(config.AUDIO_MODEL_PATH, map_location=self.device)
                if state_dict:  # Only try to load if the state dict is not empty
                    try:
                        self.audio_model.load_state_dict(state_dict)
                        print(f"Loaded audio model from {config.AUDIO_MODEL_PATH}")
                    except Exception as e:
                        print(f"Error loading audio model weights: {e}")
                else:
                    print(f"Audio model file was empty. Using random initialization.")
            else:
                print(f"Audio model file not found at {config.AUDIO_MODEL_PATH}. Using random initialization.")
            
            # Always save the models to ensure we have valid model files
            print("Saving current model states...")
            os.makedirs(config.MODEL_DIR, exist_ok=True)
            torch.save(self.image_model.state_dict(), config.IMAGE_MODEL_PATH)
            torch.save(self.audio_model.state_dict(), config.AUDIO_MODEL_PATH)
            print(f"Models saved to {config.MODEL_DIR}")
        except Exception as e:
            print(f"Error loading or creating pre-trained weights: {e}")
            print("Using random initialization for models and continuing")
        
        self.image_model.to(self.device)
        self.audio_model.to(self.device)
        self.fusion_model.to(self.device)
    
    def predict_image(self, image_tensor):
        """Make prediction using only image data."""
        image_tensor = image_tensor.to(self.device)
        with torch.no_grad():
            image_pred, attention_map = self.image_model(image_tensor)
            return image_pred, attention_map
    
    def predict_audio(self, audio_tensor):
        """Make prediction using only audio data."""
        audio_tensor = audio_tensor.to(self.device)
        with torch.no_grad():
            audio_pred, attention_weights = self.audio_model(audio_tensor)
            return audio_pred, attention_weights
    
    def predict(self, image_tensor=None, audio_tensor=None):
        """Make a prediction using available modalities."""
        image_pred, audio_pred = None, None
        image_attention, audio_attention = None, None
        
        # Get image prediction if available
        if image_tensor is not None:
            image_pred, image_attention = self.predict_image(image_tensor)
        
        # Get audio prediction if available
        if audio_tensor is not None:
            audio_pred, audio_attention = self.predict_audio(audio_tensor)
        
        # Combine modalities if both are available
        if image_pred is not None and audio_pred is not None:
            with torch.no_grad():
                combined_pred = self.fusion_model(image_pred, audio_pred)
            return combined_pred, (image_attention, audio_attention)
        elif image_pred is not None:
            return image_pred, (image_attention, None)
        elif audio_pred is not None:
            return audio_pred, (None, audio_attention)
        else:
            # If no input is provided, return zeros
            return torch.zeros(1, config.NUM_CLASSES).to(self.device), (None, None)
    
    def train(self, train_loader, num_epochs=config.NUM_EPOCHS, learning_rate=config.LEARNING_RATE):
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Optimizers
        image_optimizer = torch.optim.Adam(self.image_model.parameters(), lr=learning_rate)
        audio_optimizer = torch.optim.Adam(self.audio_model.parameters(), lr=learning_rate)
        fusion_optimizer = torch.optim.Adam(self.fusion_model.parameters(), lr=learning_rate)
        
        # Learning rate schedulers
        image_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(image_optimizer, patience=2)
        audio_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(audio_optimizer, patience=2)
        
        for epoch in range(num_epochs):
            self.image_model.train()
            self.audio_model.train()
            self.fusion_model.train()
            
            running_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                images = batch['image'].to(self.device)
                audio = batch['audio'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Zero the parameter gradients
                image_optimizer.zero_grad()
                audio_optimizer.zero_grad()
                fusion_optimizer.zero_grad()
                
                # Forward pass for image model
                image_outputs, _ = self.image_model(images)
                image_loss = criterion(image_outputs, labels)
                
                # Forward pass for audio model
                audio_outputs, _ = self.audio_model(audio)
                audio_loss = criterion(audio_outputs, labels)
                
                # Forward pass for fusion model
                fusion_outputs = self.fusion_model(image_outputs, audio_outputs)
                fusion_loss = criterion(fusion_outputs, labels)
                
                # Combined loss with weights
                total_loss = 0.3 * image_loss + 0.3 * audio_loss + 0.4 * fusion_loss
                
                # Backward and optimize
                total_loss.backward()
                image_optimizer.step()
                audio_optimizer.step()
                fusion_optimizer.step()
                
                running_loss += total_loss.item()
                
                if batch_idx % 100 == 0:
                    print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {total_loss.item():.4f} (Image: {image_loss.item():.4f}, Audio: {audio_loss.item():.4f}, Fusion: {fusion_loss.item():.4f})')
            
            # Update schedulers based on epoch loss
            epoch_loss = running_loss / len(train_loader)
            image_scheduler.step(epoch_loss)
            audio_scheduler.step(epoch_loss)
            
            print(f'Epoch {epoch} completed. Average Loss: {epoch_loss:.4f}')
        
        # Save trained models
        os.makedirs(config.MODEL_DIR, exist_ok=True)
        torch.save(self.image_model.state_dict(), config.IMAGE_MODEL_PATH)
        torch.save(self.audio_model.state_dict(), config.AUDIO_MODEL_PATH)
        
        print(f"Models saved to {config.MODEL_DIR}")

def main():
    # Dataset paths
    image_dir = config.IMAGE_DIR
    audio_dir = config.AUDIO_DIR
    csv_file = config.CSV_FILE
    
    # Create datasets
    train_dataset = BirdDataset(image_dir, audio_dir, csv_file, is_train=True)
    test_dataset = BirdDataset(image_dir, audio_dir, csv_file, is_train=False)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.BATCH_SIZE, 
        shuffle=False, 
        num_workers=config.NUM_WORKERS
    )
    
    # Initialize model
    num_classes = len(train_dataset.class_to_idx)
    model = BirdSpeciesClassifier(num_classes)
    
    # Train model
    model.train(train_loader, num_epochs=1)  # Just 1 epoch for testing

if __name__ == '__main__':
    main() 