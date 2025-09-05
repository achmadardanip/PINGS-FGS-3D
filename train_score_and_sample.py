# train_score_and_sample.py
# Apple-to-apple diffusion baselines for PINGS:
# - Train epsilon-net on 3D GMM with DDPM (VP) objective
# - Samplers: DDIM (N steps) and PF-ODE Heun (DPM-Solver-2 style, N steps)
# - Report same metrics as PINGS + projection plots + mean±std timings

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
torch.backends.cudnn.benchmark = True

# -----------------------------
# Target 3D Gaussian Mixture (same as PINGS)
# -----------------------------
class GaussianMixture3D:
    def __init__(self):
        self.pi = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32, device=device)
        self.mu = torch.tensor([[ 2.5,  0.0, -1.5],
                                [-2.0,  2.0,  1.0],
                                [ 0.0, -2.5,  2.0]], dtype=torch.float32, device=device)
        self.var = torch.tensor([[0.60**2, 0.50**2, 0.70**2],
                                 [0.45**2, 0.65**2, 0.40**2],
                                 [0.55**2, 0.40**2, 0.60**2]], dtype=torch.float32, device=device)
        self.inv_var = 1.0 / self.var
        self.log_det = torch.log(self.var).sum(dim=1)
        self.K = self.pi.shape[0]; self.dim = 3

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        comp = torch.multinomial(self.pi, num_samples=n, replacement=True)
        eps  = torch.randn((n, self.dim), device=device)
        mu   = self.mu[comp]
        std  = torch.sqrt(self.var[comp])
        return mu + eps * std

# -----------------------------
# Diffusion schedule (VP / DDPM)
# -----------------------------
def make_beta_schedule(T=1000, schedule="cosine"):
    if schedule == "linear":
        beta_start, beta_end = 1e-4, 0.02
        betas = torch.linspace(beta_start, beta_end, T, device=device)
    elif schedule == "cosine":
        # Nichol & Dhariwal cosine
        s = 0.008
        steps = T + 1
        x = torch.linspace(0, T, steps, device=device)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = betas.clamp(1e-5, 0.999)
    else:
        raise ValueError("unknown schedule")
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

# Continuous lookup for alpha_t, sigma_t, beta_t
class VPCont:
    def __init__(self, T=1000, schedule="cosine"):
        self.T = T
        self.betas, self.alphas, self.alpha_bars = make_beta_schedule(T, schedule)
        self.betas_c = self.betas.detach()
        self.alpha_bars_c = self.alpha_bars.detach()

    def at(self, t_float):
        # t_float in [0,1]
        t_scaled = t_float * (self.T - 1)
        i0 = torch.clamp(torch.floor(t_scaled).long(), 0, self.T-1)
        i1 = torch.clamp(i0 + 1, 0, self.T-1)
        w = (t_scaled - i0.float()).unsqueeze(-1)
        ab0 = self.alpha_bars_c[i0].unsqueeze(-1); ab1 = self.alpha_bars_c[i1].unsqueeze(-1)
        alpha_bar = (1 - w) * ab0 + w * ab1
        alpha = torch.sqrt(alpha_bar)
        sigma = torch.sqrt(1 - alpha_bar)
        beta = self.betas_c[i0].unsqueeze(-1)
        return alpha, sigma, beta

# -----------------------------
# Epsilon-Net (MLP) with time embedding
# -----------------------------
class TimeMLP(nn.Module):
    def __init__(self, hidden=128, depth=6, emb_dim=64):
        super().__init__()
        self.emb_dim = emb_dim
        in_dim = 3 + emb_dim
        layers = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, 3)]
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight); nn.init.zeros_(m.bias)

    def time_embed(self, t):
        # sinusoidal time features
        freqs = torch.arange(self.emb_dim//2, device=t.device, dtype=t.dtype)
        freqs = 2 * math.pi * (2.0 ** freqs)
        wt = t * freqs
        emb = torch.cat([torch.sin(wt), torch.cos(wt)], dim=-1)
        return emb

    def forward(self, x, t):
        if t.ndim == 1: t = t.unsqueeze(-1)
        emb = self.time_embed(t)
        h = torch.cat([x, emb], dim=1)
        return self.net(h)  # predict epsilon

# -----------------------------
# Training epsilon-net
# -----------------------------
def train_epsilon_net(
    epochs=20000, batch=2048, lr=1e-3, gamma=0.999,
    schedule="cosine", T=1000, ckpt="score_vp.pt", patience=3000
):
    target = GaussianMixture3D()
    vp = VPCont(T=T, schedule=schedule)
    net = TimeMLP(hidden=128, depth=6, emb_dim=64).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=gamma)

    best, noimp = float("inf"), 0
    for ep in range(1, epochs+1):
        net.train(); opt.zero_grad()
        x0 = target.sample(batch)
        t = torch.rand((batch,1), device=device).clamp_min(1e-4)
        alpha, sigma, _ = vp.at(t.squeeze(-1))
        eps = torch.randn_like(x0)
        xt  = alpha * x0 + sigma * eps

        eps_pred = net(xt, t)
        loss = F.mse_loss(eps_pred, eps)
        loss.backward(); opt.step()
        if ep % 1000 == 0: sched.step()

        if loss.item() + 1e-12 < best:
            best, noimp = loss.item(), 0
            torch.save({"model": net.state_dict()}, ckpt)
        else:
            noimp += 1

        if ep % 500 == 0:
            print(f"[{ep}] loss={loss.item():.6f} best={best:.6f} noimp={noimp}")
        if noimp >= patience:
            print(f"Early stop at {ep}. best={best:.6f}")
            break

    if os.path.exists(ckpt):
        net.load_state_dict(torch.load(ckpt, map_location=device)["model"])
    return net, vp

# -----------------------------
# Sampler 1: DDIM (deterministic, eta=0)
# -----------------------------
@torch.no_grad()
def ddim_sample(net, vp: VPCont, n=10_000, steps=50, use_amp=True):
    x = torch.randn((n,3), device=device)
    ts = torch.linspace(1.0, 0.0, steps+1, device=device)
    for i in range(steps, 0, -1):
        t  = ts[i].expand(n,1)
        tm = ts[i-1].expand(n,1)
        alpha_t, sigma_t, _ = vp.at(t.squeeze(-1))
        alpha_s, sigma_s, _ = vp.at(tm.squeeze(-1))
        if device.type == "cuda" and use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                eps = net(x, t)
        else:
            eps = net(x, t)
        x0_hat = (x - sigma_t * eps) / alpha_t
        x = alpha_s * x0_hat + sigma_s * eps
    return x

# -----------------------------
# Sampler 2: PF-ODE Heun (DPM-Solver-2 style)
# Improved eps->score conversion (apple-to-apple)
# ODE for VP: x' = -0.5*beta*x - 0.5*beta*s(x,t)
# with s(x,t) ≈ -(x - alpha*x0_hat)/sigma^2, x0_hat = (x - sigma*eps)/alpha
# -----------------------------
@torch.no_grad()
def dpm_solver2_vp(net, vp: VPCont, n=10_000, steps=10, use_amp=True):
    x = torch.randn((n,3), device=device)
    ts = torch.linspace(1.0, 0.0, steps+1, device=device)

    def drift(x_in, t_in):
        alpha, sigma, beta = vp.at(t_in.squeeze(-1))
        if device.type == "cuda" and use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                eps = net(x_in, t_in)
        else:
            eps = net(x_in, t_in)
        x0_hat = (x_in - sigma * eps) / alpha
        score  = -(x_in - alpha * x0_hat) / (sigma**2 + 1e-12)
        return -0.5 * beta * x_in - 0.5 * beta * score

    for k in range(steps):
        t0 = ts[k].expand(n,1); t1 = ts[k+1].expand(n,1)
        dt = (t1 - t0)  # negative
        f0 = drift(x, t0)
        x_pred = x + dt * f0
        f1 = drift(x_pred, t1)
        x = x + 0.5 * dt * (f0 + f1)
    return x

# -----------------------------
# Metrics (same as PINGS)
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

def mmd2(x, y, sigmas=(0.1,0.2,0.5,1.0,2.0)):
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
    return (m4.squeeze(0) / (std**4) - 3.0)

@torch.no_grad()
def stats_vs_target(x_gen, target: GaussianMixture3D):
    x_tgt = target.sample(x_gen.shape[0])
    mmd_val = mmd2(x_gen, x_tgt).item()
    mg = x_gen.mean(dim=0); mt = x_tgt.mean(dim=0)
    cg = ((x_gen - mg).T @ (x_gen - mg)) / (x_gen.shape[0]-1)
    ct = ((x_tgt - mt).T @ (x_tgt.shape[0]-1)).T  # guard; but keep original pattern below
    ct = ((x_tgt - mt).T @ (x_tgt - mt)) / (x_tgt.shape[0]-1)

    sk_g, sk_t = skewness(x_gen), skewness(x_tgt)
    ku_g, ku_t = kurtosis(x_gen), kurtosis(x_tgt)
    return {
        "MMD2": mmd_val,
        "mean_MSE": F.mse_loss(mg, mt).item(),
        "cov_MSE": F.mse_loss(cg, ct).item(),
        "skew_MSE": F.mse_loss(sk_g, sk_t).item(),
        "kurt_MSE": F.mse_loss(ku_g, ku_t).item(),
    }

def save_projection(x_gen, x_tgt, title, path):
    xg = x_gen.detach().cpu().numpy()
    xt = x_tgt.detach().cpu().numpy()
    plt.figure(figsize=(7,7))
    plt.scatter(xt[:,0], xt[:,1], s=6, alpha=0.25, label="Target (proj)", marker='o')
    plt.scatter(xg[:,0], xg[:,1], s=6, alpha=0.25, label=title+" (proj)", marker='x')
    plt.xlabel("x1"); plt.ylabel("x2"); plt.title(f"Projection (x1,x2): {title} vs Target")
    plt.legend(); plt.tight_layout()
    plt.savefig(path, dpi=160); plt.close()
    print(f"Saved {path}")

# -----------------------------
# Benchmark helper (mean ± std)
# -----------------------------
def benchmark(fn, repeat=5):
    times = []
    for _ in range(repeat):
        if device.type == "cuda": torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            fn()
        if device.type == "cuda": torch.cuda.synchronize()
        times.append(time.time() - t0)
    return float(np.mean(times)), float(np.std(times))

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    # 1) Train epsilon-net (score model)
    net, vp = train_epsilon_net(
        epochs=20000, batch=2048, lr=1e-3, gamma=0.999,
        schedule="cosine", T=1000, ckpt="score_vp.pt", patience=3000
    )
    net.eval()  # <-- ensure eval mode for sampling
    target = GaussianMixture3D()
    N = 10_000

    # 2) DDIM (50) — timing (mean±std) + stats + plot
    tmean, tstd = benchmark(lambda: ddim_sample(net, vp, n=N, steps=50, use_amp=True), repeat=5)
    with torch.inference_mode():
        x_ddim = ddim_sample(net, vp, n=N, steps=50, use_amp=True)
    s_ddim = stats_vs_target(x_ddim, target)
    s_ddim["gen_time_sec_mean_for_10k"] = tmean
    s_ddim["gen_time_sec_std_for_10k"]  = tstd
    with open("baseline_stats_ddim50.csv","w",newline="") as f:
        w=csv.writer(f); w.writerow(["metric","value"])
        for k,v in s_ddim.items(): w.writerow([k,v])
    save_projection(x_ddim, target.sample(N), "DDIM (50)", "ddim50_projection.png")
    print(f"DDIM(50) mean±std: {tmean:.6f}s ± {tstd:.6f}s")

    # 3) DPM-Solver-2 (Heun ODE) for 10 & 20 steps — timing + stats + plot
    for steps in [10, 20]:
        tmean, tstd = benchmark(lambda: dpm_solver2_vp(net, vp, n=N, steps=steps, use_amp=True), repeat=5)
        with torch.inference_mode():
            x_dpm = dpm_solver2_vp(net, vp, n=N, steps=steps, use_amp=True)
        s = stats_vs_target(x_dpm, target)
        s["gen_time_sec_mean_for_10k"] = tmean
        s["gen_time_sec_std_for_10k"]  = tstd
        with open(f"baseline_stats_dpm{steps}.csv","w",newline="") as f:
            w=csv.writer(f); w.writerow(["metric","value"])
            for k,v in s.items(): w.writerow([k,v])
        save_projection(x_dpm, target.sample(N),
                        title=f"DPM-Solver (Heun, {steps})",
                        path=f"dpm_solver{steps}_projection.png")
        print(f"DPM-Solver({steps}) mean±std: {tmean:.6f}s ± {tstd:.6f}s")
