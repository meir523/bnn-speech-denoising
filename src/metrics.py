import numpy as np
import pandas as pd
from pesq import pesq
from pystoi import stoi
from tqdm.auto import tqdm
import inference  # inference.py module

def calculate_pesq(clean_audio, degraded_audio, sr=16000):
    """
    Calculates PESQ after ensuring both arrays are exactly the same length.
    """
    # Fix length mismatch by truncating the longer array
    min_len = min(len(clean_audio), len(degraded_audio))
    clean_audio = clean_audio[:min_len]
    degraded_audio = degraded_audio[:min_len]
    
    try:
        score = pesq(sr, clean_audio, degraded_audio, 'wb')
        return score
    except Exception as e:
        print(f"PESQ failed: {e}")
        return np.nan

def calculate_stoi(clean_audio, degraded_audio, sr=16000):
    """
    Calculates STOI after ensuring both arrays are exactly the same length.
    """
    # Fix length mismatch by truncating the longer array
    min_len = min(len(clean_audio), len(degraded_audio))
    clean_audio = clean_audio[:min_len]
    degraded_audio = degraded_audio[:min_len]
    
    try:
        score = stoi(clean_audio, degraded_audio, sr, extended=False)
        return score
    except Exception as e:
        print(f"STOI failed: {e}")
        return np.nan


def evaluate_multiple_models_by_snr(dataset, models_dict, context_frames, mc_samples_inference, hop_length, device, sr=16000):
    """
    Evaluates an arbitrary dictionary of models grouped by SNR.
    models_dict format: {'Model Name': model_object}
    """
    results = {}
    
    for sample in tqdm(dataset, desc="Evaluating Dataset by SNR"):
        snr = sample['snr']
        clean_audio = sample['clean']
        noisy_audio = sample['mixed']
        
        if snr not in results:
            results[snr] = {'Noisy Baseline': {'pesq': [], 'stoi': []}}
            for model_name in models_dict.keys():
                results[snr][model_name] = {'pesq': [], 'stoi': []}
                
        # Baseline
        results[snr]['Noisy Baseline']['pesq'].append(calculate_pesq(clean_audio, noisy_audio, sr))
        results[snr]['Noisy Baseline']['stoi'].append(calculate_stoi(clean_audio, noisy_audio, sr))
        
        # Models
        for model_name, model in models_dict.items():
            # If it's a DNN, use 1 sample. If BNN, use mc_samples_inference.
            is_bnn = hasattr(model, 'get_kl_loss')
            samples_to_run = mc_samples_inference if is_bnn else 1
            
            pred = inference.run_inference(model, sample, context_frames, samples_to_run, hop_length, device)
            results[snr][model_name]['pesq'].append(calculate_pesq(clean_audio, pred, sr))
            results[snr][model_name]['stoi'].append(calculate_stoi(clean_audio, pred, sr))
            
    summary_table = []
    for snr in sorted(results.keys()):
        for model_name, scores in results[snr].items():
            summary_table.append({
                'SNR (dB)': snr,
                'Model': model_name,
                'PESQ Score': round(np.nanmean(scores['pesq']), 3),
                'STOI Score': round(np.nanmean(scores['stoi']), 3)
            })
            
    return pd.DataFrame(summary_table)