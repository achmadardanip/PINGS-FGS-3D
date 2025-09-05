# dpm_solver10_baseline.py
# Baseline: "DPM-Solver (10)" for the same 3D non-Gaussian case as PINGS
# - probability-flow ODE x'(t) = alpha(t) * score_target(x)
# - integrated with DPM-Solver-2 (Heun, order-2) using 10 steps from t=1 -> 0
# - produces the same outputs as PINGS for fair comparison:
#   * 10k samples
#   * MMD^2 and moment MSEs vs target draws
#   * wall-clock timing for generation
#   * (optional) 2D projection plot

import os, time, math, csv
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -----------------------------
# Repro & Device
# -----------------------------
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # ok for large batched ops

# -----------------------------
# Target: 3D Gaussian Mixture (same as PINGS)
# -----------------------------
class GaussianMixture3D:
    def __init__(self):
        self.pi = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32, device=device)
        self.mu = torch.tensor([
            [ 2.5,  0.0, -1.5],
            [-2.0,  2.0,  1.0],
            [ 0.0, -2.5,  2.0],
        ], dtype=torch.float32, device=device)                     # (K,3)
        self.var = torch.tensor([
            [0.60**2, 0.50**2, 0.70**2],
            [0.45**2, 0.65**2, 0.40**2],
            [0.55**2, 0.40**2, 0.60**2],
        ], dtype=torch.float32, device=device)                     # (K,3)
        self.inv_var = 1.0 / self.var
        self.log_det = torch.log(self.var).sum(dim=1)
        self.K = self.pi.shape[0]
        self.dim = 3

    def sample(self, n: int) -> torch.Tensor:
        with torch.no_grad():
            comp = torch.multinomial(self.pi, num_samples=n, replacement=True)
            eps  = torch.randn((n, self.dim), device=device)
            mu   = self.mu[comp]
            std  = torch.sqrt(self.var[comp])
            x    = mu + eps * std
        return x

    def log_prob_components(self, x: torch.Tensor) -> torch.Tensor:
        x  = x.unsqueeze(1)             # (B,1,3)
        mu = self.mu.unsqueeze(0)       # (1,K,3)
        inv= self.inv_var.unsqueeze(0)  # (1,K,3)
        diff = x - mu
        quad = (diff**2 * inv).sum(dim=2)  # (B,K)
        cst  = self.dim * math.log(2*math.pi)
        return -0.5 * (quad + self.log_det + cst)

    def responsibilities(self, x: torch.Tensor) -> torch.Tensor:
        log_Nk  = self.log_prob_components(x)          # (B,K)
        log_wNk = torch.log(self.pi).unsqueeze(0) + log_Nk
        return torch.softmax(log_wNk, dim=1)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        # ∇x log p(x) for mixture with diagonal covariances
        r   = self.responsibilities(x)                 # (B,K)
        x_  = x.unsqueeze(1)                           # (B,1,3)
        mu  = self.mu.unsqueeze(0)                     # (1,K,3)
        inv = self.inv_var.unsqueeze(0)                # (1,K,3)
        term_k = inv * (mu - x_)                       # (B,K,3)
        return (r.unsqueeze(2) * term_k).sum(dim=1)    # (B,3)

# -----------------------------
# Helpers: prior, MMD & moments (same as PINGS)
# -----------------------------
def sample_prior(n: int) -> torch.Tensor:
    return torch.randn((n, 3), device=device)

def pdist2(x, y):
    x2 = (x**2).sum(dim=1, keepdim=True)
    y2 = (y**2).sum(dim=1, keepdim=True).T
    return x2 + y2 - 2*torch.matmul(x, y.T)

def gaussian_kernel_matrix(x, y, sigmas):
    d2 = pdist2(x, y)
    k = 0.0
    for s in sigmas:
        gamma = 1.0/(2.0*s*s)
        k = k + torch.exp(-gamma * d2)
    return k

def mmd2(x, y, sigmas=(0.1, 0.2, 0.5, 1.0, 2.0)):
    Kxx = gaussian_kernel_matrix(x, x, sigmas)
    Kyy = gaussian_kernel_matrix(y, y, sigmas)
    Kxy = gaussian_kernel_matrix(x, y, sigmas)
    m, n = x.shape[0], y.shape[0]
    sum_xx = (Kxx.sum() - Kxx.diag().sum()) / (m*(m-1))
    sum_yy = (Kyy.sum() - Kyy.diag().sum()) / (n*(n-1))
    sum_xy = Kxy.mean()
    return sum_xx + sum_yy - 2*sum_xy

def central_moments(x: torch.Tensor, order: int):
    mu = x.mean(dim=0, keepdim=True)
    c  = x - mu
    if order == 2: return (c**2).mean(dim=0, keepdim=True)
    if order == 3: return (c**3).mean(dim=0, keepdim=True)
    if order == 4: return (c**4).mean(dim=0, keepdim=True)
    raise ValueError("order must be 2/3/4")

def skewness(x: torch.Tensor):
    std = x.std(dim=0, unbiased=True).clamp_min(1e-8)
    m3  = central_moments(x, 3)
    return (m3.squeeze(0) / (std**3))

def kurtosis(x: torch.Tensor):
    std = x.std(dim=0, unbiased=True).clamp_min(1e-8)
    m4  = central_moments(x, 4)
    return (m4.squeeze(0) / (std**4) - 3.0)  # excess

# -----------------------------
# DPM-Solver-2 (Heun) integrator for 10 steps
# ODE: x'(t) = alpha(t) * score(x),  t: 1 -> 0
# -----------------------------
@torch.no_grad()
def dpm_solver2_ode_sampling(
    target: GaussianMixture3D,
    n: int = 10_000,
    steps: int = 10,
    alpha_scale: float = 1.0,
    alpha_power: float = 1.0,
    use_amp: bool = True,
):
    """
    Returns samples x(t=0) by integrating the probability-flow ODE with
    a DPM-Solver-2 (Heun) scheme using 'steps' uniform steps from t=1 to t=0.
    """
    x = sample_prior(n)  # x(t=1) ~ N(0, I)
    t_grid = torch.linspace(1.0, 0.0, steps + 1, device=device)

    def alpha(t):
        return alpha_scale * torch.pow(1.0 - t, alpha_power)

    for k in range(steps):
        t0 = t_grid[k].view(1, 1)      # scalar as tensor
        t1 = t_grid[k + 1].view(1, 1)
        dt = (t1 - t0)                 # negative step

        if device.type == "cuda" and use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                f0 = alpha(t0) * target.score(x)
                x_pred = x + dt * f0
                f1 = alpha(t1) * target.score(x_pred)
                x = x + 0.5 * dt * (f0 + f1)
        else:
            f0 = alpha(t0) * target.score(x)
            x_pred = x + dt * f0
            f1 = alpha(t1) * target.score(x_pred)
            x = x + 0.5 * dt * (f0 + f1)

    return x

# -----------------------------
# Metrics & timing (same style as PINGS)
# -----------------------------
def stats_vs_target(x_gen: torch.Tensor, target: GaussianMixture3D):
    with torch.no_grad():
        x_tgt = target.sample(x_gen.shape[0])

        mmd_val = mmd2(x_gen, x_tgt).item()

        mg = x_gen.mean(dim=0); mt = x_tgt.mean(dim=0)
        cg = ((x_gen - mg).T @ (x_gen - mg)) / (x_gen.shape[0] - 1)
        ct = ((x_tgt - mt).T @ (x_tgt - mt)) / (x_tgt.shape[0] - 1)

        sk_g, sk_t = skewness(x_gen), skewness(x_tgt)
        ku_g, ku_t = kurtosis(x_gen), kurtosis(x_tgt)

        mean_mse = F.mse_loss(mg, mt).item()
        cov_mse  = F.mse_loss(cg, ct).item()
        skew_mse = F.mse_loss(sk_g, sk_t).item()
        kurt_mse = F.mse_loss(ku_g, ku_t).item()

    return {
        "MMD2": mmd_val,
        "mean_MSE": mean_mse,
        "cov_MSE": cov_mse,
        "skew_MSE": skew_mse,
        "kurt_MSE": kurt_mse,
    }

def benchmark_speed(steps=10, n=10_000, repeat=5, use_amp=True):
    target = GaussianMixture3D()
    times = []
    for _ in range(repeat):
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        _ = dpm_solver2_ode_sampling(target, n=n, steps=steps, use_amp=use_amp)
        if device.type == "cuda": torch.cuda.synchronize()
        times.append(time.time() - t0)
    return float(np.mean(times)), float(np.std(times))

def save_stats_csv(d: dict, path="dpm_solver10_stats.csv"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "value"])
        for k, v in d.items(): w.writerow([k, v])
    print(f"Saved stats to {path}")

def quick_plot_projection(x_gen, x_tgt, save_path="dpm_solver10_projection.png"):
    xg = x_gen.detach().cpu().numpy()
    xt = x_tgt.detach().cpu().numpy()
    plt.figure(figsize=(7,7))
    plt.scatter(xt[:,0], xt[:,1], s=6, alpha=0.25, label="Target (proj)", marker='o')
    plt.scatter(xg[:,0], xg[:,1], s=6, alpha=0.25, label="DPM-Solver (10) (proj)", marker='x')
    plt.xlabel("x1"); plt.ylabel("x2"); plt.title("Projection (x1,x2): DPM-Solver (10) vs Target")
    plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=160); plt.close()
    print(f"Saved projection to {save_path}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    steps = 10
    N = 10_000
    target = GaussianMixture3D()

    # Generate with DPM-Solver-2 (10 steps)
    if device.type == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    x_gen = dpm_solver2_ode_sampling(target, n=N, steps=steps, use_amp=True)
    if device.type == "cuda": torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"Wall-clock for {N} samples with DPM-Solver (steps={steps}): {elapsed:.6f} s")

    # Stats
    stats = stats_vs_target(x_gen, target)
    mean_t, std_t = benchmark_speed(steps=steps, n=N, repeat=5, use_amp=True)
    stats["gen_time_sec_mean_for_10k"] = mean_t
    stats["gen_time_sec_std_for_10k"]  = std_t
    save_stats_csv(stats, path="dpm_solver10_stats.csv")

    # Plot projection
    x_tgt = target.sample(N)
    quick_plot_projection(x_gen, x_tgt, save_path="dpm_solver10_projection.png")

    print("\n=== Summary (DPM-Solver 10) ===")
    for k, v in stats.items():
        print(f"{k}: {v:.6f}")
