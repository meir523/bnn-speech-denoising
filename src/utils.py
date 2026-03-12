import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import librosa
import librosa.display


##### Set Seed #####
def set_seed(seed=42):
    '''Set the seed for the random number generators'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        # Force cuDNN to be deterministic for fully reproducible BNN sampling
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


##### Plot Learning Curve #####

def plot_learning_curve(history, title='Learning Curve'):
    '''Plot the learning curve'''
    # Plot training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Training Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()



def calculate_sparsity(model, threshold=1e-2):
    """Calculates the percentage of weights close to zero (sparsity)."""
    zero_weights = 0
    total_weights = 0
    
    for name, param in model.named_parameters():
        if 'weight' in name and 'rho' not in name:
            zero_weights += torch.sum(torch.abs(param) < threshold).item()
            total_weights += param.numel()
            
    if total_weights == 0:
        return 0.0
    
    sparsity_percentage = (zero_weights / total_weights) * 100
    return sparsity_percentage



def plot_bnn_uncertainty(model, sample, context_frames, device, sr=16000, hop_length=256, mc_samples=30):
    """
    Generates a 4-panel plot showing the clean, noisy, predicted spectrograms,
    and the predictive variance (uncertainty) of the BNN.
    """
    model.eval()
    
    noisy_mag = sample['mixed_mag']
    clean_mag = sample['clean_mag']
    snr_level = sample['snr']
    
    noisy_mag_t = noisy_mag.T
    time_frames, freq_bins = noisy_mag_t.shape
    padded_mag = np.pad(noisy_mag_t, ((context_frames, context_frames), (0, 0)), mode='constant')
    
    windows = [padded_mag[i : i + 2 * context_frames + 1].flatten() for i in range(time_frames)]
    inputs_tensor = torch.tensor(np.array(windows), dtype=torch.float32).to(device)
    
    sampled_mags = []
    with torch.no_grad():
        for _ in range(mc_samples):
            pred_mag = model(inputs_tensor).cpu().numpy()
            sampled_mags.append(pred_mag)
            
    sampled_mags = np.array(sampled_mags)
    mean_pred = np.mean(sampled_mags, axis=0).T
    var_pred = np.var(sampled_mags, axis=0).T
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Clean
    img1 = librosa.display.specshow(librosa.amplitude_to_db(clean_mag, ref=np.max), 
                                    y_axis='linear', x_axis='time', sr=sr, hop_length=hop_length, ax=axes[0, 0])
    axes[0, 0].set_title('Clean Speech Spectrogram')
    fig.colorbar(img1, ax=axes[0, 0], format="%+2.0f dB")
    
    # 2. Noisy
    img2 = librosa.display.specshow(librosa.amplitude_to_db(noisy_mag, ref=np.max), 
                                    y_axis='linear', x_axis='time', sr=sr, hop_length=hop_length, ax=axes[0, 1])
    axes[0, 1].set_title(f'Noisy Speech Spectrogram (SNR: {snr_level}dB)')
    fig.colorbar(img2, ax=axes[0, 1], format="%+2.0f dB")
    
    # 3. Mean Prediction
    img3 = librosa.display.specshow(librosa.amplitude_to_db(mean_pred, ref=np.max), 
                                    y_axis='linear', x_axis='time', sr=sr, hop_length=hop_length, ax=axes[1, 0])
    axes[1, 0].set_title('BNN Mean Prediction (MMSE)')
    fig.colorbar(img3, ax=axes[1, 0], format="%+2.0f dB")
    
    # 4. Variance (Uncertainty)
    img4 = librosa.display.specshow(var_pred, 
                                    y_axis='linear', x_axis='time', sr=sr, hop_length=hop_length, ax=axes[1, 1], cmap='magma')
    axes[1, 1].set_title('BNN Predictive Variance (Uncertainty)')
    fig.colorbar(img4, ax=axes[1, 1], label='Variance')
    
    plt.tight_layout()
    plt.show()


##### Plot OOD Comparison #####

def plot_ood_comparison(dnn_model, bnn_model, context_frames, device, sr=16000, hop_length=256, mc_samples=30):
    """
    Generates an Out-of-Distribution (OOD) sine wave mixed with white noise,
    runs both DNN and BNN, and plots a comparison highlighting BNN uncertainty.
    """
    # 1. Generate OOD Audio (1000Hz Sine wave + White Noise)
    time_duration = 3.0 
    t = np.linspace(0, time_duration, int(sr * time_duration), endpoint=False)
    ood_clean = 0.5 * np.sin(2 * np.pi * 1000 * t) 
    ood_noise = np.random.normal(0, 0.5, ood_clean.shape)
    ood_mixed = ood_clean + ood_noise
    
    mixed_stft = librosa.stft(ood_mixed, n_fft=512, hop_length=hop_length)
    noisy_mag = np.abs(mixed_stft)
    
    # 2. Prepare for Inference
    noisy_mag_t = noisy_mag.T
    time_frames, freq_bins = noisy_mag_t.shape
    padded_mag = np.pad(noisy_mag_t, ((context_frames, context_frames), (0, 0)), mode='constant')
    
    windows = [padded_mag[i : i + 2 * context_frames + 1].flatten() for i in range(time_frames)]
    inputs_tensor = torch.tensor(np.array(windows), dtype=torch.float32).to(device)
    
    # 3. DNN Inference (Single Forward Pass)
    dnn_model.eval()
    with torch.no_grad():
        dnn_pred_mag = dnn_model(inputs_tensor).cpu().numpy().T
        
    # 4. BNN Inference (Monte Carlo Sampling)
    bnn_model.eval()
    sampled_mags = []
    with torch.no_grad():
        for _ in range(mc_samples):
            pred_mag = bnn_model(inputs_tensor).cpu().numpy()
            sampled_mags.append(pred_mag)
            
    sampled_mags = np.array(sampled_mags)
    bnn_mean_pred = np.mean(sampled_mags, axis=0).T
    bnn_var_pred = np.var(sampled_mags, axis=0).T
    
    # 5. Plotting the 2x2 Grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Top-Left: Noisy OOD Input
    img1 = librosa.display.specshow(librosa.amplitude_to_db(noisy_mag, ref=np.max), 
                                    y_axis='linear', x_axis='time', sr=sr, hop_length=hop_length, ax=axes[0, 0])
    axes[0, 0].set_title('Out-of-Distribution Input (Sine + White Noise)')
    fig.colorbar(img1, ax=axes[0, 0], format="%+2.0f dB")
    
    # Top-Right: DNN Prediction
    img2 = librosa.display.specshow(librosa.amplitude_to_db(dnn_pred_mag, ref=np.max), 
                                    y_axis='linear', x_axis='time', sr=sr, hop_length=hop_length, ax=axes[0, 1])
    axes[0, 1].set_title('DNN Prediction (Confident but wrong)')
    fig.colorbar(img2, ax=axes[0, 1], format="%+2.0f dB")
    
    # Bottom-Left: BNN Mean Prediction
    img3 = librosa.display.specshow(librosa.amplitude_to_db(bnn_mean_pred, ref=np.max), 
                                    y_axis='linear', x_axis='time', sr=sr, hop_length=hop_length, ax=axes[1, 0])
    axes[1, 0].set_title('BNN Mean Prediction (MMSE)')
    fig.colorbar(img3, ax=axes[1, 0], format="%+2.0f dB")
    
    # Bottom-Right: BNN Predictive Variance
    img4 = librosa.display.specshow(bnn_var_pred, 
                                    y_axis='linear', x_axis='time', sr=sr, hop_length=hop_length, ax=axes[1, 1], cmap='magma')
    axes[1, 1].set_title('BNN Predictive Variance (Detects OOD anomaly)')
    fig.colorbar(img4, ax=axes[1, 1], label='Variance')
    
    plt.tight_layout()
    plt.show()



##### Plot Weight Distributions #####

def plot_weight_distributions(dnn_model, bnn_model):
    # Extract weights from the first linear layer of the DNN
    dnn_weights = dnn_model.fc1.weight.detach().cpu().numpy().flatten()
    
    # Extract mean weights (mu) from the first Bayesian layer of the BNN
    bnn_mu = bnn_model.fc1.weight_mu.detach().cpu().numpy().flatten()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # DNN Histogram
    ax1.hist(dnn_weights, bins=100, color='blue', alpha=0.7)
    ax1.set_title('DNN Weights (Point Estimates)')
    ax1.set_xlabel('Weight Value')
    ax1.set_ylabel('Frequency')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # BNN Histogram
    ax2.hist(bnn_mu, bins=100, color='orange', alpha=0.7)
    ax2.set_title('BNN Mean Weights (Laplace Prior Sparsity)')
    ax2.set_xlabel('Weight Mean Value')
    ax2.set_ylabel('Frequency')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()