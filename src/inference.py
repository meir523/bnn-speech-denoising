import torch
import numpy as np
import librosa


##### Reconstruct Audio #####

def reconstruct_audio(magnitude, phase, hop_length=256):
    '''Reconstruct the audio from the magnitude and phase'''
    # Combine magnitude and phase to form the complex STFT matrix
    stft_matrix = magnitude * np.exp(1j * phase)
    
    # Perform Inverse STFT to generate the time-domain waveform
    reconstructed_signal = librosa.istft(stft_matrix, hop_length=hop_length)
    
    return reconstructed_signal


##### Run Inference #####   

def run_inference(model, sample, context_frames, mc_samples=20, hop_length=256, device='cuda'):
    model.eval()
    
    noisy_mag = sample['mixed_mag']
    noisy_phase = sample['mixed_phase']
    
    noisy_mag_t = noisy_mag.T
    time_frames, freq_bins = noisy_mag_t.shape
    padded_mag = np.pad(noisy_mag_t, ((context_frames, context_frames), (0, 0)), mode='constant')
    
    windows = []
    for i in range(time_frames):
        window = padded_mag[i : i + 2 * context_frames + 1].flatten()
        windows.append(window)
        
    inputs_tensor = torch.tensor(np.array(windows), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        # Check if model is BNN by looking for our custom method
        if hasattr(model, 'get_kl_loss'):
            # Bayesian Neural Network: Perform Monte Carlo sampling
            sampled_mags = []
            for _ in range(mc_samples):
                pred_mag = model(inputs_tensor).cpu().numpy()
                sampled_mags.append(pred_mag)
            
            # MMSE: Calculate the mean over all stochastic forward passes
            predicted_mag_t = np.mean(sampled_mags, axis=0)
        else:
            # Deterministic DNN: Single forward pass
            predicted_mag_t = model(inputs_tensor).cpu().numpy()
            
    predicted_mag = predicted_mag_t.T
    
    # Clean up any potential NaNs or Infs from stochastic extreme edge cases
    predicted_mag = np.nan_to_num(predicted_mag, nan=0.0, posinf=0.0, neginf=0.0)    
    predicted_audio = reconstruct_audio(predicted_mag, noisy_phase, hop_length=hop_length)
    return predicted_audio