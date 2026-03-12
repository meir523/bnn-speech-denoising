# src/train.py
import torch
import torch.nn as nn
import torch.optim as optim


##### Train DNN #####
def train_dnn(model, train_loader, val_loader, epochs=10, learning_rate=1e-3, device='cuda'):
    # Define the loss function for regression (Mean Squared Error) and the optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Move the model to the target hardware (GPU)
    model.to(device)
    
    # Dictionary to track losses for plotting later
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            # Move data to GPU
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients to prevent accumulation
            optimizer.zero_grad()
            
            # Forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(inputs)
            
            # Calculate the batch loss
            loss = criterion(outputs, targets)
            
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            
            # Perform a single optimization step
            optimizer.step()
            
            # Accumulate the loss
            running_loss += loss.item() * inputs.size(0)
            
        # Calculate average loss over the entire training epoch
        epoch_train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(epoch_train_loss)
        
        # Validation phase
        # Set the model to evaluation mode (disables dropout/batchnorm if used)
        model.eval()
        val_loss = 0.0
        
        # Disable gradient calculation for memory efficiency and speed
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                
        # Calculate average validation loss
        epoch_val_loss = val_loss / len(val_loader.dataset)
        history['val_loss'].append(epoch_val_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        
    return history


##### Train BNN #####

def train_bnn(model, train_loader, val_loader, epochs=15, learning_rate=1e-3, max_beta=0.005, mc_samples=3, device='cuda'):  
    '''Train the Bayesian Neural Network''' 
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.to(device)
    history = {'train_loss': [], 'val_loss': [], 'mse_loss': [], 'kl_loss': []}

    total_batches = len(train_loader)
    
    for epoch in range(epochs):
        # Set the model to training mode
        model.train()
        running_loss = running_mse = running_kl = 0.0
        
        # KL Annealing: linearly increase the KL weight from 0 to max_beta over the first 70% of epochs
        # This prevents Posterior Collapse and forces the network to learn to denoise first.
        annealing_factor = min(1.0, epoch / (epochs * 0.7))
        current_beta = max_beta * annealing_factor
        kl_weight = current_beta / total_batches
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            
            # Monte Carlo sampling during training to stabilize gradients
            batch_mse = 0.0
            batch_kl = 0.0
            
            for _ in range(mc_samples):
                outputs = model(inputs)
                batch_mse += criterion(outputs, targets)
                batch_kl += model.get_kl_loss()
                
            # Average the losses over the MC samples
            mse_loss = batch_mse / mc_samples
            kl_loss = batch_kl / mc_samples
            
            # ELBO Loss Calculation
            loss = mse_loss + kl_weight * kl_loss
            
            loss.backward()
            # Clip the gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_mse += mse_loss.item() * inputs.size(0)
            running_kl += kl_loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_mse = running_mse / len(train_loader.dataset)
        epoch_kl = running_kl / len(train_loader.dataset)
        
        history['train_loss'].append(epoch_loss)
        history['mse_loss'].append(epoch_mse)
        history['kl_loss'].append(epoch_kl)
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_mse_sum = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Single pass is often sufficient for fast validation tracking
                outputs = model(inputs)
                mse_loss = criterion(outputs, targets)
                kl_loss = model.get_kl_loss()
                loss = mse_loss + kl_weight * kl_loss
                
                val_loss += loss.item() * inputs.size(0)
                val_mse_sum += mse_loss.item() * inputs.size(0)
                
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_mse = val_mse_sum / len(val_loader.dataset)

        history['val_loss'].append(epoch_val_loss)
        if 'val_mse' not in history:
            history['val_mse'] = []
        history['val_mse'].append(epoch_val_mse)
        
        print(f"Epoch [{epoch+1}/{epochs}] | ELBO: {epoch_loss:.4f} (MSE: {epoch_mse:.4f}, KL(unscaled): {epoch_kl:.4f}) | Beta: {current_beta:.5f}")
        
    return history