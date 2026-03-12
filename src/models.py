import torch
import torch.nn as nn
import torch.nn.functional as F
import math

##### Speech DNN #####
class SpeechDNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=257):
        super(SpeechDNN, self).__init__()
        
        # Define the network architecture
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu1 = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.ReLU()
        
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = nn.ReLU()
        
        # The output layer returns the cleaned magnitude spectrum
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # Using ReLU at the output since magnitude cannot be negative
        self.relu_out = nn.ReLU()

    def forward(self, x):
        # Forward pass through the hidden layers
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        # Final output projection
        x = self.relu_out(self.fc_out(x))
        return x

    

##### Bayesian Linear Layer #####

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_sigma=0.1, prior_type='gaussian'):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma
        self.prior_type = prior_type.lower()

        # Learnable parameters for the mean (mu) of the weights and biases
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        
        # Learnable parameters for the standard deviation, parameterized as rho to ensure positivity
        self.weight_rho = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias_rho = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize the parameters
        self.reset_parameters()
        
        # Placeholder for the KL divergence calculated during the forward pass
        self.kl_loss = 0.0

    def reset_parameters(self):
        # Initialize means similarly to standard PyTorch linear layers
        nn.init.kaiming_normal_(self.weight_mu, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(self.bias_mu)
        
        # Initialize rho to a small value (represents a small initial variance)
        nn.init.constant_(self.weight_rho, -3.0)
        nn.init.constant_(self.bias_rho, -3.0)

    def forward(self, x):
        # Transform rho to standard deviation using numerically stable softplus
        weight_sigma = F.softplus(self.weight_rho) + 1e-8
        bias_sigma = F.softplus(self.bias_rho) + 1e-8
        
        # Sample epsilon from a normal distribution N(0, prior_sigma)
        epsilon_weight = torch.randn_like(weight_sigma)
        epsilon_bias = torch.randn_like(bias_sigma)
        
        # Apply the reparameterization trick: w = mu + sigma * epsilon
        weight = self.weight_mu + weight_sigma * epsilon_weight
        bias = self.bias_mu + bias_sigma * epsilon_bias
        
        # Calculate KL divergence based on the selected prior
        if self.prior_type == 'gaussian':
            self.kl_loss = self._kl_divergence_gaussian(self.weight_mu, weight_sigma) + \
                           self._kl_divergence_gaussian(self.bias_mu, bias_sigma)
        elif self.prior_type == 'laplace':
            self.kl_loss = self._kl_divergence_laplace(weight, self.weight_mu, weight_sigma) + \
                           self._kl_divergence_laplace(bias, self.bias_mu, bias_sigma)
        else:
            raise ValueError(f"Unsupported prior_type: {self.prior_type}")        
        
        # Perform the linear operation with the sampled weights
        return F.linear(x, weight, bias)


    def _kl_divergence_gaussian(self, mu, sigma):
        # Analytical KL for N(mu, sigma) || N(0, prior_sigma)
        kl = 0.5 * torch.sum(2 * torch.log(self.prior_sigma / sigma) + \
                             (sigma**2 + mu**2) / (self.prior_sigma**2) - 1.0)
        return kl
        
    def _kl_divergence_laplace(self, sample, mu, sigma):
        # Monte Carlo approximation of KL divergence for Laplace prior: log(q(w)) - log(p(w))
        # q(w) is the Gaussian posterior N(mu, sigma^2)
        log_q = -0.5 * math.log(2 * math.pi) - torch.log(sigma) - 0.5 * ((sample - mu) / sigma)**2
        
        # p(w) is the Laplace prior Laplace(0, b) where b is prior_sigma
        b = self.prior_sigma
        log_p = -math.log(2 * b) - torch.abs(sample) / b
        
        return torch.sum(log_q - log_p)


##### Speech BNN #####

class SpeechBNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=257, prior_sigma=0.1, prior_type='gaussian'):
        super(SpeechBNN, self).__init__()
        
        # Define Bayesian layers matching the architecture of SpeechDNN
        self.fc1 = BayesianLinear(input_dim, hidden_dim, prior_sigma, prior_type)
        self.relu1 = nn.ReLU()
        
        self.fc2 = BayesianLinear(hidden_dim, hidden_dim, prior_sigma, prior_type)
        self.relu2 = nn.ReLU()
        
        self.fc3 = BayesianLinear(hidden_dim, hidden_dim, prior_sigma, prior_type)
        self.relu3 = nn.ReLU()
        
        self.fc_out = BayesianLinear(hidden_dim, output_dim, prior_sigma, prior_type)
        self.relu_out = nn.ReLU()

    def forward(self, x):
        # Forward pass samples new weights each time and accumulates KL loss implicitly
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        x = self.relu_out(self.fc_out(x))
        return x

    def get_kl_loss(self):
        # Sum the KL divergence from all Bayesian layers to be used in the final loss function
        return self.fc1.kl_loss + self.fc2.kl_loss + self.fc3.kl_loss + self.fc_out.kl_loss