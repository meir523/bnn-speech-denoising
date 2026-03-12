import os
import glob
import random
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


##### Data Loading #####

def get_timit_files(data_dir, num_speakers=10, files_per_speaker=5):
    # Build path to the reduced TIMIT dataset
    timit_path = os.path.join(data_dir, "TIMITreduced")
    
    # Collect all subdirectories
    speaker_dirs = [os.path.join(timit_path, d) for d in os.listdir(timit_path) 
                    if os.path.isdir(os.path.join(timit_path, d))]
    
    # Sort to ensure reproducibility, then select the required amount
    speaker_dirs.sort()
    selected_speakers = speaker_dirs[:num_speakers]
    
    selected_files = []
    for speaker_dir in selected_speakers:
        # Find all wav files for the current speaker
        wav_files = glob.glob(os.path.join(speaker_dir, "*.wav"))
        wav_files.sort()
        
        # Select the required number of files
        selected_files.extend(wav_files[:files_per_speaker])
        
    return selected_files

def get_wind_files(data_dir):
    # Build path to the wind noise dataset
    wind_path = os.path.join(data_dir, "wind")
    
    # Find and return all wind wav files
    wind_files = glob.glob(os.path.join(wind_path, "*.wav"))
    return wind_files

def load_and_mix(speech_path, wind_path, target_snr_db, sr=16000):
    # Load the audio signals using librosa
    speech, _ = librosa.load(speech_path, sr=sr)
    noise, _ = librosa.load(wind_path, sr=sr)
    
    # Calculate the power of both signals
    speech_power = np.mean(speech ** 2)
    noise_power = np.mean(noise ** 2)
    
    # Handle edge cases where signal or noise is completely silent
    if noise_power == 0 or speech_power == 0:
        return speech, speech, noise
        
    # Determine the target noise power based on the desired SNR
    target_noise_power = speech_power / (10 ** (target_snr_db / 10))
    scaling_factor = np.sqrt(target_noise_power / noise_power)
    
    # Apply the scaling factor to the noise
    scaled_noise = noise * scaling_factor
    
    # Ensure both arrays are the exact same length by truncating the longer one
    min_len = min(len(speech), len(scaled_noise))
    speech = speech[:min_len]
    scaled_noise = scaled_noise[:min_len]
    
    # Combine the clean speech and the scaled noise
    mixed_signal = speech + scaled_noise
    
    return mixed_signal, speech, scaled_noise

   
##### STFT Features #####

def compute_stft_features(signal, n_fft=512, hop_length=256):
    '''Compute the Short-Time Fourier Transform of the signal'''
    stft_matrix = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    
    # Extract magnitude and phase components
    magnitude = np.abs(stft_matrix)
    phase = np.angle(stft_matrix)
    
    return magnitude, phase

def process_dataset_features(dataset, n_fft=512, hop_length=256):
    """
    Computes and appends STFT magnitude and phase features for each item in the dataset.
    """
    # Iterate over the dataset to compute and append spectrogram features
    for item in dataset:
        clean_mag, clean_phase = compute_stft_features(item['clean'], n_fft, hop_length)
        mixed_mag, mixed_phase = compute_stft_features(item['mixed'], n_fft, hop_length)
        
        # Store the extracted features back into the dataset dictionary
        item['clean_mag'] = clean_mag
        item['clean_phase'] = clean_phase
        item['mixed_mag'] = mixed_mag
        item['mixed_phase'] = mixed_phase
        
    return dataset


##### Split Dataset #####

def split_dataset_by_speaker(dataset_list, train_ratio=0.7, val_ratio=0.15, seed=42):
    """
    Splits the dataset into train, validation, and test sets strictly by speaker ID
    to prevent data leakage.
    """
    random.seed(seed)
    
    # Extract unique speakers from the dataset
    all_speakers = list(set([item['speaker_id'] for item in dataset_list]))
    all_speakers.sort() # Sort for reproducibility
    random.shuffle(all_speakers)
    
    num_speakers = len(all_speakers)
    train_end = int(train_ratio * num_speakers)
    val_end = train_end + int(val_ratio * num_speakers)
    
    # Allocate speakers to each split
    train_speakers = set(all_speakers[:train_end])
    val_speakers = set(all_speakers[train_end:val_end])
    test_speakers = set(all_speakers[val_end:])
    
    # Route the actual data samples based on their speaker ID
    train_data = [item for item in dataset_list if item['speaker_id'] in train_speakers]
    val_data = [item for item in dataset_list if item['speaker_id'] in val_speakers]
    test_data = [item for item in dataset_list if item['speaker_id'] in test_speakers]
    
    print(f"Split complete. Speakers: {len(train_speakers)} Train, {len(val_speakers)} Val, {len(test_speakers)} Test.")
    print(f"Samples: {len(train_data)} Train, {len(val_data)} Val, {len(test_data)} Test.")
    
    return train_data, val_data, test_data



##### Speech Dataset - PyTorch Dataset Adapter #####

class SpeechDataset(Dataset):
    """
    A PyTorch dataset wrapper for speech datasets, supporting context windows.
    """
    def __init__(self, dataset_list, context_frames=3):
        self.inputs = []
        self.targets = []
        
        for item in dataset_list:
            mixed_mag = item['mixed_mag']
            clean_mag = item['clean_mag']
            
            mixed_mag = mixed_mag.T
            clean_mag = clean_mag.T
            
            time_frames, freq_bins = mixed_mag.shape
            padded_mixed = np.pad(mixed_mag, ((context_frames, context_frames), (0, 0)), mode='constant')
            
            for i in range(time_frames):
                window = padded_mixed[i : i + 2 * context_frames + 1]
                self.inputs.append(window.flatten())
                self.targets.append(clean_mag[i])
        
        self.inputs = torch.tensor(np.array(self.inputs), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)
        
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

def create_data_loaders(train_dataset_list, val_dataset_list, test_dataset_list, batch_size=128, context_frames=3):
    """
    Returns PyTorch DataLoaders for train, validation, and test datasets.
    """
    train_dataset = SpeechDataset(train_dataset_list, context_frames)
    val_dataset = SpeechDataset(val_dataset_list, context_frames)
    test_dataset = SpeechDataset(test_dataset_list, context_frames)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


##### Overfitting Loaders - single speaker #####

def create_overfitting_loaders(dataset_list, batch_size=128, context_frames=3, speaker_id=None):
    """
    Creates dataloaders for a single speaker to intentionally cause overfitting.
    Takes 80% of their files for training and 20% for testing.
    """
    if speaker_id is None:
        # Default to the first speaker found
        speaker_id = dataset_list[0]['speaker_id']
        
    # Isolate data for this single speaker
    speaker_data = [item for item in dataset_list if item['speaker_id'] == speaker_id]
    
    # Split chronologically (e.g., 80% train, 20% test)
    split_idx = int(len(speaker_data) * 0.8)
    train_data = speaker_data[:split_idx]
    test_data = speaker_data[split_idx:]
    
    print(f"Isolated Speaker {speaker_id} for Overfitting Test: {len(train_data)} train samples, {len(test_data)} test samples.")
    
    train_dataset = SpeechDataset(train_data, context_frames)
    test_dataset = SpeechDataset(test_data, context_frames)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader
 