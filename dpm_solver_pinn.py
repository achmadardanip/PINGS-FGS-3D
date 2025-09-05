"""
Physics-Informed Neural Network (PINN) with DPM-Solver for Generative Modeling

This module implements a novel approach combining Physics-Informed Neural Networks
with DPM-Solver for learning and sampling from complex probability distributions.

Architecture Overview:
1. GaussianMixture3D: Target distribution (3D Gaussian mixture model)
2. VelocityPINN: Neural network learning the velocity field f(x,t) -> dx/dt
3. DPMSolver: Second-order ODE solver using the learned velocity field
4. Evaluation: Comprehensive metrics including MMD, Wasserstein distance, moments

Mathematical Framework:
- ODE Path: x_t = (1-t)*z0 + t*x1, where t ∈ [0,1]
- Velocity: dx/dt = f(x,t) = x1 - z0 (ground truth during training)
- Integration: From prior N(0,I) at t=0 to data distribution at t=1

Dependencies:
- PyTorch: Neural networks and automatic differentiation
- NumPy: Numerical operations
- Matplotlib: Visualization
- SciPy: Statistical computations (Wasserstein distance)
"""


import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

# ===============================================
#  Setup: Reproducibility & Device Configuration
# ===============================================

# Set random seeds for reproducible results across runs
torch.manual_seed(42)
np.random.seed(42)

# Configure computational device (CUDA if available, else CPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===============================================
#  Target Distribution & Prior Sampling
# ===============================================

class GaussianMixture3D:
    """
    3D Gaussian Mixture Model representing the target data distribution p(x).
    
    This class defines a mixture of 3 Gaussian components in 3D space with
    different mixing weights, means, and variances. It serves as the target
    distribution that our generative model aims to learn and sample from.
    
    Components:
    - Component 1 (50%): Centered at [2.5, 0.0, -1.5] with moderate spread
    - Component 2 (30%): Centered at [-2.0, 2.0, 1.0] with varied spread
    - Component 3 (20%): Centered at [0.0, -2.5, 2.0] with balanced spread
    
    Mathematical form: p(x) = Σ π_i * N(x | μ_i, Σ_i)
    """
    
    def __init__(self):
        """Initialize the 3-component Gaussian mixture model parameters."""
        
        # Mixing weights (probabilities): sum to 1.0
        self.pi = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32, device=device)
        
        # Mean vectors for each component [3 components × 3 dimensions]
        self.mu = torch.tensor([
            [ 2.5,  0.0, -1.5],  # Component 1: upper-right-back
            [-2.0,  2.0,  1.0],  # Component 2: lower-left-front  
            [ 0.0, -2.5,  2.0],  # Component 3: center-bottom-front
        ], dtype=torch.float32, device=device)
        
        # Variance parameters for each component [3 components × 3 dimensions]
        # Using diagonal covariance matrices for computational efficiency
        self.var = torch.tensor([
            [0.60**2, 0.50**2, 0.70**2],  # Component 1 variances
            [0.45**2, 0.65**2, 0.40**2],  # Component 2 variances
            [0.55**2, 0.40**2, 0.60**2],  # Component 3 variances
        ], dtype=torch.float32, device=device)
        
        # Dimensionality of the space
        self.dim = 3

    def sample(self, n: int) -> torch.Tensor:
        """
        Generate n samples from the Gaussian mixture distribution.
        
        Sampling process:
        1. Select mixture component for each sample using multinomial sampling
        2. Generate Gaussian noise
        3. Transform noise using component-specific mean and variance
        
        Args:
            n (int): Number of samples to generate
            
        Returns:
            torch.Tensor: Generated samples of shape [n, 3]
        """
        with torch.no_grad():
            # Step 1: Choose mixture component for each sample
            # multinomial returns indices of selected components
            comp = torch.multinomial(self.pi, num_samples=n, replacement=True)
            
            # Step 2: Generate standard normal noise
            eps = torch.randn((n, self.dim), device=device)
            
            # Step 3: Transform using selected component parameters
            mu_selected = self.mu[comp]          # Select means: [n, 3]
            var_selected = self.var[comp]        # Select variances: [n, 3]
            std_selected = torch.sqrt(var_selected)  # Convert to std dev
            
            # Apply affine transformation: x = μ + σ * ε
            x = mu_selected + eps * std_selected
            
        return x

def sample_prior(n: int, dim: int = 3) -> torch.Tensor:
    """
    Sample from the prior distribution N(0, I).
    
    The prior is a standard multivariate Gaussian centered at origin.
    This serves as the starting point (t=0) for the ODE integration.
    
    Args:
        n (int): Number of samples to generate
        dim (int): Dimensionality of samples (default: 3)
        
    Returns:
        torch.Tensor: Prior samples of shape [n, dim]
    """
    return torch.randn((n, dim), device=device)

# ===============================================
#  Physics-Informed Neural Network (PINN) Architecture
# ===============================================

class VelocityPINN(nn.Module):
    """
    Physics-Informed Neural Network for learning the velocity field f(x,t) -> dx/dt.
    
    Architecture:
    - Input: [x₁, x₂, x₃, t] (4D: 3D position + 1D time)
    - Hidden: Multiple fully connected layers with SiLU activation
    - Output: [v₁, v₂, v₃] (3D velocity vector)
    
    Physics Constraint:
    The network learns to approximate the true velocity field that defines
    the ODE: dx/dt = f(x,t), where the ODE path connects prior to target distribution.
    
    Training Data:
    - Input: (x_t, t) where x_t = (1-t)*z₀ + t*x₁
    - Target: dx/dt = x₁ - z₀ (analytical derivative of linear path)
    """
    
    def __init__(self, hidden_dim=128, depth=6):
        """
        Initialize the velocity field neural network.
        
        Args:
            hidden_dim (int): Width of hidden layers (default: 128)
            depth (int): Number of hidden layers (default: 6)
        """
        super().__init__()
        
        # Network dimensions
        in_dim = 4   # [x₁, x₂, x₃, t]
        out_dim = 3  # [v₁, v₂, v₃]
        
        # Create layer dimensions list
        dims = [in_dim] + [hidden_dim] * depth + [out_dim]
        
        # Build fully connected layers
        self.layers = nn.ModuleList([
            nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)
        ])
        
        # SiLU (Swish) activation: f(x) = x * sigmoid(x)
        # Often performs better than ReLU for smooth function approximation
        self.activation = nn.SiLU()

        # Xavier (Glorot) initialization for stable training
        # Maintains variance of activations across layers
        with torch.no_grad():
            for layer in self.layers:
                nn.init.xavier_normal_(layer.weight, gain=1.0)
                nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: compute velocity field f(x,t).
        
        Handles different time tensor shapes for flexibility:
        - Scalar tensor: t = tensor(0.5) -> broadcast to [N, 1]  
        - Vector tensor: t = tensor([N]) -> reshape to [N, 1]
        - Matrix tensor: t = tensor([[N, 1]]) -> use as-is
        
        Args:
            x (torch.Tensor): Position tensor of shape [N, 3]
            t (torch.Tensor): Time tensor (scalar, [N], or [N, 1])
            
        Returns:
            torch.Tensor: Velocity vectors of shape [N, 3]
        """
        # Handle different time tensor shapes for robust inference
        if t.ndim == 0:  # Scalar tensor case (common in ODE solvers)
            # Broadcast scalar to match batch size
            t = t.expand(x.shape[0], 1)
        elif t.ndim == 1:  # 1D vector case
            # Add dimension for concatenation
            t = t.unsqueeze(1)
        # t.ndim == 2 case ([N, 1]) requires no modification
        
        # Concatenate position and time: [N, 3] + [N, 1] -> [N, 4]
        h = torch.cat([x, t], dim=1)
        
        # Forward pass through hidden layers with activation
        for layer in self.layers[:-1]:
            h = self.activation(layer(h))
        
        # Final layer without activation (linear output)
        return self.layers[-1](h)

# ===============================================
#  PINN Training Pipeline
# ===============================================

def train_velocity_pinn(
    target_dist,
    epochs=15000,
    batch_size=2048,
    lr=1e-3,
    gamma=0.995,
    save_path="velocity_pinn.pt"
):
    """
    Train the velocity PINN to learn the ODE velocity field.
    
    Training Strategy:
    1. Sample pairs (z₀, x₁) from prior and target distributions
    2. Create ODE path: x_t = (1-t)*z₀ + t*x₁ with random t
    3. Compute true velocity: dx/dt = x₁ - z₀ (analytical)
    4. Train network to predict this velocity: f(x_t, t) ≈ x₁ - z₀
    5. Use MSE loss for regression training
    
    Optimization:
    - Adam optimizer with exponential learning rate decay
    - Best model checkpointing based on validation loss
    - Scheduled learning rate reduction for fine-tuning
    
    Args:
        target_dist: Target distribution object with .sample() method
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        lr (float): Initial learning rate
        gamma (float): Learning rate decay factor
        save_path (str): Path to save the best model
        
    Returns:
        VelocityPINN: Trained neural network model
    """
    print("--- Training Velocity PINN ---")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {lr}")
    
    # Initialize network and optimization components
    net = VelocityPINN().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    # Track best model for checkpointing
    best_loss = float('inf')

    # Training loop
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # Generate training data pairs
        # z₀: samples from prior N(0,I) at t=0
        z0 = sample_prior(batch_size)
        # x₁: samples from target distribution p(x) at t=1  
        x1 = target_dist.sample(batch_size)

        # Sample random times along the ODE path [0, 1]
        t = torch.rand(batch_size, 1, device=device)

        # Define the straight-line ODE path (linear interpolation)
        # x_t = (1-t)*z₀ + t*x₁
        # This creates a valid ODE connecting prior to target
        xt = (1 - t) * z0 + t * x1

        # Analytical ground truth velocity field
        # d/dt[(1-t)*z₀ + t*x₁] = -z₀ + x₁ = x₁ - z₀
        true_velocity = x1 - z0

        # Network prediction for velocity at (x_t, t)
        predicted_velocity = net(xt, t)

        # Regression loss: minimize MSE between predicted and true velocity
        loss = F.mse_loss(predicted_velocity, true_velocity)
        
        # Backpropagation and parameter update
        loss.backward()
        optimizer.step()

        # Learning rate scheduling (every 1000 epochs)
        if epoch % 1000 == 0:
            scheduler.step()

        # Model checkpointing: save best model based on loss
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(net.state_dict(), save_path)

        # Progress reporting
        if epoch % 500 == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}, LR: {current_lr:.6f}")

    print(f"--- Training Complete ---")
    print(f"Best Loss: {best_loss:.6f}")
    
    # Load the best saved model
    print(f"Loading best model from {save_path}")
    net.load_state_dict(torch.load(save_path, map_location=device))
    
    return net

# ===============================================
#  DPM-Solver: Second-Order ODE Integration
# ===============================================

class DPMSolver:
    """
    Second-order DPM-Solver for ODE integration using learned velocity fields.
    
    This solver implements a multi-step numerical integration scheme that can
    work with any velocity function (analytical or neural network-based).
    
    Mathematical Foundation:
    - ODE: dx/dt = f(x,t) where f is the velocity field
    - Integration: Numerical solution from t=0 to t=1
    - Order: Second-order accuracy using previous derivative information
    
    Algorithm:
    1. First step: Euler method x_{n+1} = x_n + h*f(x_n, t_n)
    2. Subsequent steps: Adams-Bashforth-like scheme using previous derivatives
       x_{n+1} = x_n + h*(1.5*f(x_n, t_n) - 0.5*f(x_{n-1}, t_{n-1}))
    
    Advantages:
    - Higher accuracy than first-order methods
    - Stable for smooth velocity fields
    - Efficient computation with minimal memory overhead
    """
    
    def __init__(self, velocity_fn):
        """
        Initialize the DPM solver with a velocity function.
        
        Args:
            velocity_fn: Function f(x,t) -> dx/dt
                        Can be analytical function or neural network
        """
        self.velocity_fn = velocity_fn

    @torch.inference_mode()
    def solve(self, x0: torch.Tensor, n_steps: int, t_start: float = 0.0, t_end: float = 1.0):
        """
        Solve the ODE dx/dt = f(x,t) from t_start to t_end.
        
        Integration Process:
        1. Create uniform time grid from t_start to t_end
        2. Initialize with starting condition x0
        3. For each time step:
           - Compute current velocity f(x_current, t_current)  
           - Update position using second-order formula
           - Store previous derivative for next iteration
        
        Args:
            x0 (torch.Tensor): Initial conditions at t=t_start, shape [N, dim]
            n_steps (int): Number of integration steps
            t_start (float): Starting time (default: 0.0)  
            t_end (float): Ending time (default: 1.0)
            
        Returns:
            torch.Tensor: Solution at t=t_end, shape [N, dim]
        """
        print(f"Running DPM-Solver-2 with {n_steps} steps from t={t_start} to t={t_end}")
        
        # Create uniform time discretization
        ts = torch.linspace(t_start, t_end, n_steps + 1, device=device)
        
        # Initialize solution and derivative storage
        x = x0.clone()  # Current solution
        d_prev = None   # Previous derivative (for second-order accuracy)

        # Integration loop
        for i in range(n_steps):
            t_curr = ts[i]      # Current time
            t_next = ts[i+1]    # Next time
            h = t_next - t_curr # Step size

            # Evaluate velocity field at current state
            d_curr = self.velocity_fn(x, t_curr)

            if d_prev is None:
                # First step: Use first-order Euler method
                # x_{n+1} = x_n + h * f(x_n, t_n)
                x = x + h * d_curr
            else:
                # Subsequent steps: Use second-order Adams-Bashforth scheme
                # x_{n+1} = x_n + h * (1.5*f(x_n, t_n) - 0.5*f(x_{n-1}, t_{n-1}))
                # This uses current and previous derivatives for higher accuracy
                x = x + h * (1.5 * d_curr - 0.5 * d_prev)
            
            # Store current derivative for next iteration
            d_prev = d_curr
        
        return x

# ===============================================
#  Evaluation Metrics and Statistical Analysis
# ===============================================

def pdist2(x, y):
    """
    Compute pairwise squared Euclidean distances between two sets of points.
    
    Uses efficient matrix operations to avoid explicit loops:
    ||x_i - y_j||² = ||x_i||² + ||y_j||² - 2⟨x_i, y_j⟩
    
    Args:
        x (torch.Tensor): First set of points [m, d]
        y (torch.Tensor): Second set of points [n, d]
        
    Returns:
        torch.Tensor: Distance matrix [m, n] where entry (i,j) = ||x_i - y_j||²
    """
    # Compute squared norms: ||x_i||² for all i
    x2 = (x**2).sum(dim=1, keepdim=True)  # [m, 1]
    # Compute squared norms: ||y_j||² for all j  
    y2 = (y**2).sum(dim=1, keepdim=True).T # [1, n]
    # Compute inner products: 2⟨x_i, y_j⟩ for all i,j
    xy = 2 * torch.matmul(x, y.T)          # [m, n]
    
    # Combine using broadcasting: [m, 1] + [1, n] - [m, n] -> [m, n]
    return x2 + y2 - xy

def gaussian_kernel_matrix(x, y, sigmas):
    """
    Compute Gaussian kernel matrix with multiple bandwidth parameters.
    
    The Gaussian (RBF) kernel is: k(x,y) = exp(-||x-y||²/(2σ²))
    Multiple bandwidths provide scale-invariant distance measurements.
    
    Args:
        x (torch.Tensor): First point set [m, d]
        y (torch.Tensor): Second point set [n, d] 
        sigmas (tuple): Bandwidth parameters (σ₁, σ₂, ..., σₖ)
        
    Returns:
        torch.Tensor: Combined kernel matrix [m, n]
    """
    # Compute all pairwise squared distances
    d2 = pdist2(x, y)
    
    # Sum kernels across all bandwidth scales
    k = 0.0
    for sigma in sigmas:
        gamma = 1.0 / (2.0 * sigma * sigma)  # Gaussian bandwidth parameter
        k = k + torch.exp(-gamma * d2)       # Add scaled kernel
    
    return k

def mmd2(x, y, sigmas=(0.1, 0.2, 0.5, 1.0, 2.0)):
    """
    Compute Maximum Mean Discrepancy (MMD²) between two distributions.
    
    MMD is a kernel-based statistical test for comparing distributions:
    MMD²[F,p,q] = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]
    
    Where:
    - X,X' ~ p (samples from first distribution)
    - Y,Y' ~ q (samples from second distribution)  
    - k(·,·) is a positive definite kernel
    
    Properties:
    - MMD² = 0 iff p = q (when using characteristic kernels)
    - Larger values indicate greater distribution mismatch
    - Scale-invariant when using multiple bandwidths
    
    Args:
        x (torch.Tensor): Samples from first distribution [m, d]
        y (torch.Tensor): Samples from second distribution [n, d]
        sigmas (tuple): Gaussian kernel bandwidths
        
    Returns:
        float: MMD² statistic (non-negative)
    """
    # Compute kernel matrices
    Kxx = gaussian_kernel_matrix(x, x, sigmas)  # k(X,X')
    Kyy = gaussian_kernel_matrix(y, y, sigmas)  # k(Y,Y')  
    Kxy = gaussian_kernel_matrix(x, y, sigmas)  # k(X,Y)
    
    m, n = x.shape[0], y.shape[0]
    
    # E[k(X,X')] with X ≠ X' (exclude diagonal terms)
    sum_xx = (Kxx.sum() - Kxx.diag().sum()) / (m * (m - 1))
    
    # E[k(Y,Y')] with Y ≠ Y' (exclude diagonal terms)
    sum_yy = (Kyy.sum() - Kyy.diag().sum()) / (n * (n - 1))
    
    # E[k(X,Y)] (all cross terms)
    sum_xy = Kxy.mean()
    
    # MMD² = E[k(X,X')] + E[k(Y,Y')] - 2E[k(X,Y)]
    return sum_xx + sum_yy - 2 * sum_xy

def central_moments(x: torch.Tensor, order: int):
    """
    Compute central moments of a tensor about its mean.
    
    Central moment of order k: E[(X - μ)^k] where μ = E[X]
    
    Args:
        x (torch.Tensor): Input data [n, d]
        order (int): Moment order (3 for skewness, 4 for kurtosis)
        
    Returns:
        torch.Tensor: Central moments [1, d]
    """
    # Compute sample mean
    mu = x.mean(dim=0, keepdim=True)
    
    # Center the data
    c = x - mu
    
    if order == 3:
        return (c**3).mean(dim=0, keepdim=True)
    if order == 4:
        return (c**4).mean(dim=0, keepdim=True)
    
    raise ValueError("Order must be 3 or 4")

def skewness(x: torch.Tensor):
    """
    Compute the skewness (third standardized moment) of a distribution.
    
    Skewness = E[(X-μ)³] / σ³
    - Measures asymmetry of the distribution
    - Skewness = 0: symmetric (like normal distribution)  
    - Skewness > 0: right tail is longer
    - Skewness < 0: left tail is longer
    
    Args:
        x (torch.Tensor): Input data [n, d]
        
    Returns:
        torch.Tensor: Skewness values [d]
    """
    # Compute standard deviation (with Bessel's correction)
    std = x.std(dim=0, unbiased=True).clamp_min(1e-8)
    
    # Compute third central moment
    m3 = central_moments(x, 3)
    
    # Normalize by cubed standard deviation
    return (m3.squeeze(0) / (std**3))

def kurtosis(x: torch.Tensor):
    """
    Compute the excess kurtosis (fourth standardized moment minus 3).
    
    Excess Kurtosis = E[(X-μ)⁴] / σ⁴ - 3
    - Measures tail heaviness relative to normal distribution
    - Kurtosis = 0: same tail behavior as normal (mesokurtic)
    - Kurtosis > 0: heavier tails than normal (leptokurtic)  
    - Kurtosis < 0: lighter tails than normal (platykurtic)
    
    Args:
        x (torch.Tensor): Input data [n, d]
        
    Returns:
        torch.Tensor: Excess kurtosis values [d]
    """
    # Compute standard deviation (with Bessel's correction)
    std = x.std(dim=0, unbiased=True).clamp_min(1e-8)
    
    # Compute fourth central moment
    m4 = central_moments(x, 4)
    
    # Normalize and subtract 3 (excess kurtosis)
    return (m4.squeeze(0) / (std**4) - 3.0)

def stats_vs_target(x_gen: torch.Tensor, target: GaussianMixture3D):
    """
    Compute comprehensive statistics comparing generated samples to target distribution.
    
    Metrics computed:
    - MMD²: Distribution similarity using kernel methods
    - Mean MSE: First moment matching error
    - Covariance MSE: Second moment matching error  
    - Skewness MSE: Third moment matching error
    - Kurtosis MSE: Fourth moment matching error
    
    Args:
        x_gen (torch.Tensor): Generated samples [n, d]
        target (GaussianMixture3D): Target distribution object
        
    Returns:
        dict: Dictionary of computed statistics
    """
    with torch.no_grad():
        # Sample from target distribution for comparison
        x_tgt = target.sample(x_gen.shape[0])

        # Distribution similarity metric
        mmd_val = mmd2(x_gen, x_tgt).item()

        # Moment matching metrics
        # First moments (means)
        mg, mt = x_gen.mean(dim=0), x_tgt.mean(dim=0)
        mean_mse = F.mse_loss(mg, mt).item()
        
        # Second moments (covariances)  
        cg = ((x_gen - mg).T @ (x_gen - mg)) / (x_gen.shape[0] - 1)
        ct = ((x_tgt - mt).T @ (x_tgt - mt)) / (x_tgt.shape[0] - 1)
        cov_mse = F.mse_loss(cg, ct).item()
        
        # Third moments (skewness)
        sk_g, sk_t = skewness(x_gen), skewness(x_tgt)
        skew_mse = F.mse_loss(sk_g, sk_t).item()
        
        # Fourth moments (kurtosis)
        ku_g, ku_t = kurtosis(x_gen), kurtosis(x_tgt)
        kurt_mse = F.mse_loss(ku_g, ku_t).item()

    return {
        "MMD2": mmd_val,
        "mean_MSE": mean_mse,
        "cov_MSE": cov_mse,
        "skew_MSE": skew_mse,
        "kurt_MSE": kurt_mse,
    }

def evaluate_generative_model(x_gen: torch.Tensor, target_dist: GaussianMixture3D):
    """
    Comprehensive evaluation of generative model performance.
    
    MODIFIED VERSION: This now returns a dictionary of key metrics for experiment logging.
    """
    print("\n" + "="*50)
    print("      Generative Model Performance Evaluation")
    print("="*50)
    
    N = x_gen.shape[0]
    x_tgt = target_dist.sample(N)

    # Core distributional metrics
    mmd_val = mmd2(x_gen, x_tgt).item()
    
    # Moment matching analysis
    mg, mt = x_gen.mean(0), x_tgt.mean(0)
    mean_mse = F.mse_loss(mg, mt).item()
    
    cg, ct = torch.cov(x_gen.T), torch.cov(x_tgt.T)
    cov_mse = F.mse_loss(cg, ct).item()

    # Effective Damping Ratio Calculation
    m_eff = torch.trace(ct).item()
    epsilon = 1e-8
    c_eff = 1.0 / (mmd_val + epsilon)
    k_eff = 1.0
    damping_ratio = c_eff / (2 * np.sqrt(m_eff * k_eff))

    # Display results
    print(f"Metrics based on {N} samples:")
    print(f"  - MMD²: {mmd_val:.4e}")
    print(f"  - Mean MSE: {mean_mse:.4e}")
    print(f"  - Cov MSE: {cov_mse:.4e}")
    print(f"  - Effective Damping Ratio (ζ): {damping_ratio:.4f}")
    print("="*50 + "\n")
    
    # Return key metrics for logging
    return {
        "damping_ratio_zeta": damping_ratio,
        "mean_MSE": mean_mse,
        "cov_MSE": cov_mse,
        "MMD2": mmd_val,
        "x_tgt_for_plotting": x_tgt # Pass this along for final plots
    }

def save_stats_csv(stats_dict: dict, path="pinn_dmp_solver_stats.csv"):
    """
    Save statistics dictionary to CSV file for further analysis.
    
    Creates a structured CSV with metric names and values that can be
    easily imported into spreadsheet applications or analysis tools.
    
    Args:
        stats_dict (dict): Dictionary containing metric names and values
        path (str): Output file path for the CSV
    """
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        # Write header row
        writer.writerow(["metric", "value"])
        # Write data rows
        for metric, value in stats_dict.items():
            writer.writerow([metric, value])
    print(f"Saved stats to {path}")

def plot_marginal_distributions(x_gen_np, x_tgt_np, save_path="dmp_solver_marginals.png"):
    """
    Create marginal distribution comparison plots for each dimension.
    
    Generates histograms comparing the marginal distributions of generated
    vs target samples across all dimensions. This helps identify which
    dimensions are well-captured and which may need improvement.
    
    Args:
        x_gen_np (np.ndarray): Generated samples [n, d]  
        x_tgt_np (np.ndarray): Target samples [n, d]
        save_path (str): Path to save the plot image
    """
    dims = x_gen_np.shape[1]
    fig, axes = plt.subplots(1, dims, figsize=(5 * dims, 4.5), constrained_layout=True)
    fig.suptitle("Marginal Distribution Comparison", fontsize=16, y=1.05)

    for i in range(dims):
        ax = axes[i]
        
        # Target distribution histogram (filled)
        ax.hist(x_tgt_np[:, i], bins=50, density=True, alpha=0.7, 
               label='Target', color='royalblue')
        
        # Generated distribution histogram (outline only)  
        ax.hist(x_gen_np[:, i], bins=50, density=True, alpha=0.7,
               label='Generated', color='darkorange', histtype='step', linewidth=2)
        
        ax.set_title(f'Dimension {i+1}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Saved marginals plot to {save_path}")

def plot_projection(x_gen_np, x_tgt_np, n_steps_solver, save_path="dmp_solver_3d_projection.png"):
    """
    Create 2D projection scatter plot comparing generated vs target samples.
    
    Projects the 3D distributions onto the first two dimensions (x₁, x₂)
    to visualize the overall distribution structure and clustering behavior.
    
    Args:
        x_gen_np (np.ndarray): Generated samples [n, 3]
        x_tgt_np (np.ndarray): Target samples [n, 3]  
        n_steps_solver (int): Number of solver steps (for plot title)
        save_path (str): Path to save the plot image
    """
    plt.figure(figsize=(8, 8))
    
    # Target samples (circles with transparency)
    plt.scatter(x_tgt_np[:,0], x_tgt_np[:,1], s=8, alpha=0.3, 
               label="Target (proj)", marker='o', color='royalblue')
    
    # Generated samples (x markers with transparency)
    plt.scatter(x_gen_np[:,0], x_gen_np[:,1], s=8, alpha=0.3,
               label=f"DPM-Solver-{n_steps_solver} (proj)", marker='x', color='darkorange')
    
    plt.title("Projection (x₁, x₂): DPM-Solver vs Target", fontsize=14)
    plt.xlabel('x₁')
    plt.ylabel('x₂') 
    plt.legend()
    plt.grid(True, ls='--', alpha=0.6)
    plt.axis('equal')  # Equal scaling for both axes
    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.show()
    print(f"Saved projection plot to {save_path}")

# ===============================================
#  Main Execution Pipeline
# ===============================================

if __name__ == "__main__":
    """
    MODIFIED Main execution pipeline for running experiments.
    
    Pipeline Overview:
    1. Initialize and train the PINN (once).
    2. Define a set of experiments (different solver steps).
    3. Loop through each experiment:
       a. Generate samples using the specific number of steps.
       b. Evaluate performance and collect metrics.
    4. Save all collected metrics to a single CSV file.
    5. Generate plots for the final, highest-quality run.
    """
    
    print("="*60)
    print("     PINN-DPM Generative Modeling EXPERIMENT PIPELINE")  
    print("="*60)
    
    # Step 1: Initialize target and Train the PINN (Done only once)
    print("Step 1 & 2: Initializing and Training Velocity PINN...")
    target = GaussianMixture3D()
    velocity_net = train_velocity_pinn(
        target_dist=target, epochs=15000, batch_size=2048, 
        lr=1e-3, gamma=0.995, save_path="velocity_pinn.pt"
    )
    velocity_net.eval()
    pinn_velocity_field = lambda x, t: velocity_net(x, t)
    dmp_solver = DPMSolver(velocity_fn=pinn_velocity_field)
    print("PINN training complete. Starting experiments.")

    # Step 2: Define experiments and data storage
    N_SAMPLES = 10000
    # We will test the model with different numbers of integration steps
    solver_steps_to_test = [5, 10, 15, 20, 25, 50] 
    experiment_results = []

    # Step 3: Loop through each experiment configuration
    for n_steps in solver_steps_to_test:
        print("\n" + "="*60)
        print(f"      RUNNING EXPERIMENT: {n_steps} SOLVER STEPS")
        print("="*60)
        
        # Generate samples for the current configuration
        z_prior = sample_prior(N_SAMPLES)
        x_generated = dmp_solver.solve(x0=z_prior, n_steps=n_steps)
        
        # Evaluate the generated samples and get metrics
        stats = evaluate_generative_model(x_generated, target)
        
        # Add experiment-specific info to the results
        stats['n_integration_steps'] = n_steps
        experiment_results.append(stats)

    # Step 4: Save all collected results to a CSV file
    csv_filename = "pinn_dpm_experiment_results.csv"
    print(f"\nAll experiments complete. Saving results to {csv_filename}...")
    
    # Get headers from the keys of the first result dictionary (excluding plot data)
    headers = [key for key in experiment_results[0].keys() if key != 'x_tgt_for_plotting']
    
    with open(csv_filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for result in experiment_results:
            # Create a copy to write, excluding the tensor data
            row_to_write = {k: v for k, v in result.items() if k in headers}
            writer.writerow(row_to_write)
            
    print(f"Successfully saved experiment data to {csv_filename}")

    # Step 5: Generate visualization plots for the BEST run (last one)
    print("\nGenerating visualization plots for the final run...")
    final_run_stats = experiment_results[-1]
    x_gen_np = x_generated.detach().cpu().numpy() # from the last run
    x_tgt_np = final_run_stats['x_tgt_for_plotting'].detach().cpu().numpy()
    
    plot_marginal_distributions(x_gen_np, x_tgt_np)
    plot_projection(x_gen_np, x_tgt_np, solver_steps_to_test[-1])

    print("\n" + "="*60)
    print("           PIPELINE COMPLETED SUCCESSFULLY")
    print("="*60)