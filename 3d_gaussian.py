# PINGS: Fast Generative Sampling of Non-Gaussian 3D Densities (final)
import os, time, math, csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# -----------------------------
# Repro & Device
# -----------------------------
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # autotune kernels (safe for MLP)

# -----------------------------
# Target: 3D Mixture of Gaussians
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
        r   = self.responsibilities(x)                 # (B,K)
        x_  = x.unsqueeze(1)                           # (B,1,3)
        mu  = self.mu.unsqueeze(0)                     # (1,K,3)
        inv = self.inv_var.unsqueeze(0)                # (1,K,3)
        term_k = inv * (mu - x_)                       # (B,K,3)
        return (r.unsqueeze(2) * term_k).sum(dim=1)    # (B,3)

# -----------------------------
# Prior z ~ N(0,I) in R^3
# -----------------------------
def sample_prior(n: int) -> torch.Tensor:
    return torch.randn((n, 3), device=device)

# -----------------------------
# PINGS network g(t,z)->x
# -----------------------------
class PINGSNet(nn.Module):
    def __init__(self, hidden=128, depth=6):
        super().__init__()
        in_dim, out_dim = 1+3, 3  # (t,z)->x
        dims = [in_dim] + [hidden]*depth + [out_dim]
        self.fcs = nn.ModuleList([nn.Linear(dims[i], dims[i+1]) for i in range(len(dims)-1)])
        self.act = nn.Tanh()
        with torch.no_grad():
            for fc in self.fcs[:-1]:
                nn.init.xavier_normal_(fc.weight, gain=1.0); nn.init.zeros_(fc.bias)
            nn.init.xavier_normal_(self.fcs[-1].weight, gain=1.0); nn.init.zeros_(self.fcs[-1].bias)

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        h = torch.cat([t, z], dim=1)
        for fc in self.fcs[:-1]:
            h = self.act(fc(h))
        return self.fcs[-1](h)

# -----------------------------
# MMD utils
# -----------------------------
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

# -----------------------------
# Extra moments: skew & kurt
# -----------------------------
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
    return (m4.squeeze(0) / (std**4) - 3.0)  # excess kurtosis

# -----------------------------
# Train PINGS
# -----------------------------
def train_pings(
    epochs=20000,
    batch=2048,
    n_target_batch=2048,
    lr=1e-3,
    gamma=0.999,
    patience=3000,
    save_path="pings_3d.pt",
    lambda_bc=1.0,
    lambda_mmd=2.0,
    lambda_mom=0.1,
    lambda_phys=0.5,
    alpha_scale=1.0,
    alpha_power=1.0,
    use_target_score=True,   # True: residual uses target score (stable). False: proxy no-score.
):
    target = GaussianMixture3D()
    net = PINGSNet(hidden=128, depth=6).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

    best, noimp = float("inf"), 0

    for ep in range(1, epochs+1):
        net.train(); opt.zero_grad()

        # (1) BC at t=1: g(1,z) ~ z
        z_bc = sample_prior(batch)
        t1   = torch.ones((batch,1), device=device)
        x1   = net(t1, z_bc)
        loss_bc = F.mse_loss(x1, z_bc)

        # (2) Dist. match at t=0: MMD + moment (mean/cov)
        z0 = sample_prior(batch); t0 = torch.zeros((batch,1), device=device)
        x0 = net(t0, z0)
        y  = target.sample(n_target_batch)

        loss_mmd = mmd2(x0, y)
        mg, mt = x0.mean(0, keepdim=True), y.mean(0, keepdim=True)
        cg = ((x0 - mg).T @ (x0 - mg)) / (x0.shape[0]-1)
        ct = ((y  - mt).T @ (y  - mt)) / (y.shape[0]-1)
        loss_mom = F.mse_loss(mg, mt) + F.mse_loss(cg, ct)

        # (3) Physics-like residual along t
        zt = sample_prior(batch)
        tt = torch.rand((batch,1), device=device, requires_grad=True)
        xt = net(tt, zt)
        gt = torch.autograd.grad(outputs=xt, inputs=tt,
                                 grad_outputs=torch.ones_like(xt),
                                 create_graph=True, retain_graph=True, only_inputs=True)[0]  # (B,3)
        alpha_t = alpha_scale * torch.pow(1.0 - tt, alpha_power)
        if use_target_score:
            score_tgt = target.score(xt)
            residual  = gt - alpha_t * score_tgt
        else:
            # no explicit score: contractive proxy near t=0 (relies on dist loss at t=0)
            residual  = gt  # encourages small dt dynamics; identity enforced by BC at t=1
        loss_phys = (residual**2).mean()

        loss = lambda_bc*loss_bc + lambda_mmd*loss_mmd + lambda_mom*loss_mom + lambda_phys*loss_phys
        loss.backward(); opt.step()

        if ep % 1000 == 0: sched.step()

        if loss.item() + 1e-12 < best:
            best, noimp = loss.item(), 0
            torch.save({"epoch": ep, "model": net.state_dict(), "best_loss": best}, save_path)
        else:
            noimp += 1

        if ep % 200 == 0:
            print(f"[{ep:05d}] loss={loss.item():.6f} | bc={loss_bc.item():.6f} | "
                  f"mmd={loss_mmd.item():.6f} | mom={loss_mom.item():.6f} | "
                  f"phys={loss_phys.item():.6f} | lr={opt.param_groups[0]['lr']:.3e} | "
                  f"best={best:.6f} | noimp={noimp}")

        if noimp >= patience:
            print(f"Early stopping at epoch {ep}. Best loss={best:.6f}")
            break

    if os.path.exists(save_path):
        ckpt = torch.load(save_path, map_location=device)
        net.load_state_dict(ckpt["model"])
        print(f"Loaded best checkpoint @epoch={ckpt['epoch']} (best_loss={ckpt['best_loss']:.6f})")

    return net

# -----------------------------
# Evaluation
# -----------------------------
def generate_samples(net, n=10000, t=0.0, use_amp=True):
    net.eval()
    with torch.inference_mode():
        z  = sample_prior(n)
        tt = torch.full((n,1), float(t), device=device)
        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                x = net(tt, z)
        else:
            x = net(tt, z)
    return x

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

def benchmark_speed(net, n=10000, repeat=5, use_amp=True):
    times = []
    net.eval()
    z  = sample_prior(n)
    tt = torch.zeros((n,1), device=device)
    for _ in range(repeat):
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            if use_amp and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    _ = net(tt, z)
            else:
                _ = net(tt, z)
        if device.type == "cuda": torch.cuda.synchronize()
        times.append(time.time() - t0)
    return float(np.mean(times)), float(np.std(times))

def save_stats_csv(d: dict, path="pings_3d_stats.csv"):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "value"])
        for k,v in d.items(): w.writerow([k, v])
    print(f"Saved stats to {path}")

def quick_plot_projection(x_gen, x_tgt, save_path="pings_3d_projection.png"):
    xg = x_gen.detach().cpu().numpy()
    xt = x_tgt.detach().cpu().numpy()
    plt.figure(figsize=(7,7))
    plt.scatter(xt[:,0], xt[:,1], s=6, alpha=0.25, label="Target (proj)", marker='o')
    plt.scatter(xg[:,0], xg[:,1], s=6, alpha=0.25, label="PINGS (proj)", marker='x')
    plt.xlabel("x1"); plt.ylabel("x2"); plt.title("Projection (x1,x2): PINGS vs Target")
    plt.legend(); plt.tight_layout()
    plt.savefig(save_path, dpi=160); plt.close()
    print(f"Saved projection to {save_path}")

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    net = train_pings(
        epochs=20000,
        batch=2048,
        n_target_batch=2048,
        lr=1e-3,
        gamma=0.999,
        patience=3000,
        save_path="pings_3d.pt",
        lambda_bc=1.0,
        lambda_mmd=2.0,   # set 0.0 if ingin "purist" moments-only
        lambda_mom=0.1,
        lambda_phys=0.5,
        alpha_scale=1.0,
        alpha_power=1.0,
        use_target_score=True,  # True (stabil). False = proxy no-score
    )

    N = 10_000
    target = GaussianMixture3D()
    x_gen  = generate_samples(net, n=N, t=0.0, use_amp=True)
    x_tgt  = target.sample(N)

    stats = stats_vs_target(x_gen, target)
    mean_t, std_t = benchmark_speed(net, n=N, repeat=5, use_amp=True)
    stats["gen_time_sec_mean_for_10k"] = mean_t
    stats["gen_time_sec_std_for_10k"]  = std_t
    save_stats_csv(stats, path="pings_3d_stats.csv")

    quick_plot_projection(x_gen, x_tgt, save_path="pings_3d_projection.png")

    print("\n=== Summary ===")
    for k, v in stats.items():
        print(f"{k}: {v:.6f}")
