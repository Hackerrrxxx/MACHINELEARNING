import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import librosa
import numpy as np
from torchvision import transforms

class BirdDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with the following structure:
                root_dir/
                    ├── images/
                    │   ├── species1/
                    │   │   ├── image1.jpg
                    │   │   ├── image2.jpg
                    │   │   └── ...
                    │   ├── species2/
                    │   └── ...
                    └── audio/
                        ├── species1/
                        │   ├── audio1.wav
                        │   ├── audio2.wav
                        │   └── ...
                        ├── species2/
                        └── ...
            transform (callable, optional): Optional transform to be applied on images
        """
        self.root_dir = root_dir
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ])
        
        # Get all species folders
        self.species_folders = sorted(os.listdir(os.path.join(root_dir, 'images')))
        self.species_to_idx = {species: idx for idx, species in enumerate(self.species_folders)}
        
        # Create list of all image and audio pairs
        self.data_pairs = []
        for species in self.species_folders:
            species_idx = self.species_to_idx[species]
            
            # Get all images for this species
            image_dir = os.path.join(root_dir, 'images', species)
            audio_dir = os.path.join(root_dir, 'audio', species)
            
            # Get all image files
            image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # Get all audio files
            audio_files = [f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3'))]
            
            # Create pairs of image and audio files
            for img_file in image_files:
                # Find corresponding audio file (if exists)
                base_name = os.path.splitext(img_file)[0]
                matching_audio = [af for af in audio_files if os.path.splitext(af)[0] == base_name]
                
                if matching_audio:
                    audio_file = matching_audio[0]
                    self.data_pairs.append({
                        'image_path': os.path.join(image_dir, img_file),
                        'audio_path': os.path.join(audio_dir, audio_file),
                        'species_idx': species_idx
                    })
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        data_pair = self.data_pairs[idx]
        
        # Load and transform image
        image = Image.open(data_pair['image_path']).convert('RGB')
        image = self.transform(image)
        
        # Load and process audio
        y, sr = librosa.load(data_pair['audio_path'])
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        audio = torch.FloatTensor(mel_spec_db)
        
        return {
            'image': image,
            'audio': audio,
            'label': data_pair['species_idx']
        } 