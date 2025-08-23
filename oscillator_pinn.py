import os
import math
import csv
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Repro & Device
# =========================
torch.manual_seed(42)
np.random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# PINN Model Definition
# =========================
class PINN(nn.Module):
    """
    Fully-connected network for PINN: input (z, xi) -> output x
    """
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList(
            [nn.Linear(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        )
        # Xavier init
        with torch.no_grad():
            for lin in self.linears:
                nn.init.xavier_normal_(lin.weight, gain=1.0)
                nn.init.zeros_(lin.bias)

    def forward(self, x):
        # Expect x shape [N, 2] = (z, xi)
        if x.ndim != 2 or x.shape[1] != 2:
            raise ValueError(f"Input tensor must have shape [N, 2], got {tuple(x.shape)}")
        a = x
        for i in range(len(self.linears) - 1):
            a = self.activation(self.linears[i](a))
        return self.linears[-1](a)

# =========================
# Analytical Solution
# =========================
def analytical_solution(z, xi):
    """
    ODE: d^2x/dz^2 + 2*xi*dx/dz + x = 0
    IC: x(0)=0.7, dx/dz(0)=1.2
    Underdamped case (xi < 1)
    Supports tensor z, xi on same device.
    """
    if not torch.is_tensor(z):
        z = torch.tensor(z, dtype=torch.float32, device=device)
    if not torch.is_tensor(xi):
        xi = torch.tensor(xi, dtype=torch.float32, device=device)

    if xi.ndim == 0:
        xi = xi.expand_as(z)
    elif xi.shape != z.shape:
        xi = xi.expand_as(z)

    # safe sqrt for omega_d = sqrt(1 - xi^2)
    omega_d = torch.sqrt(torch.clamp(1.0 - xi**2, min=1e-12))

    C1 = torch.tensor(0.7, dtype=z.dtype, device=device)
    C2 = (1.2 + xi * C1) / omega_d

    exp_term = torch.exp(-xi * z)
    cos_term = torch.cos(omega_d * z)
    sin_term = torch.sin(omega_d * z)
    return exp_term * (C1 * cos_term + C2 * sin_term)

# =========================
# Training
# =========================
def train_pinn(
    epochs=20000,
    n_ic=100,
    n_col=2000,
    lr=1e-3,
    gamma=0.99,
    patience=1500,
    ckpt_path="pinn_dho_best.pt",
):
    layers = [2, 32, 32, 32, 32, 1]
    pinn = PINN(layers).to(device)

    optimizer = torch.optim.Adam(pinn.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    best_loss = math.inf
    epochs_no_improve = 0

    for epoch in range(epochs):
        pinn.train()
        optimizer.zero_grad()

        # -------------------------
        # Regenerate training points each epoch
        # -------------------------
        # IC points (z=0), xi ~ U[0.1, 0.4]
        z_ic = torch.zeros((n_ic, 1), device=device, requires_grad=True)
        xi_ic = (torch.rand((n_ic, 1), device=device) * 0.3 + 0.1)
        ic_input = torch.cat([z_ic, xi_ic], dim=1)
        x_ic_true = torch.full((n_ic, 1), 0.7, device=device)

        # Collocation points: z ~ U[0,20], xi ~ U[0.1,0.4]
        z_col = (torch.rand((n_col, 1), device=device) * 20.0).requires_grad_(True)
        xi_col = (torch.rand((n_col, 1), device=device) * 0.3 + 0.1)
        col_input = torch.cat([z_col, xi_col], dim=1)

        # -------------------------
        # Loss: Initial Conditions
        # -------------------------
        x_pred_ic = pinn(ic_input)
        loss_ic_val = torch.mean((x_pred_ic - x_ic_true) ** 2)

        dxdz_pred_ic = torch.autograd.grad(
            outputs=x_pred_ic,
            inputs=z_ic,
            grad_outputs=torch.ones_like(x_pred_ic),
            create_graph=True
        )[0]
        loss_ic_deriv = torch.mean((dxdz_pred_ic - 1.2) ** 2)

        # -------------------------
        # Loss: Physics residual
        # -------------------------
        x_pred_col = pinn(col_input)

        dxdz_col = torch.autograd.grad(
            outputs=x_pred_col,
            inputs=z_col,
            grad_outputs=torch.ones_like(x_pred_col),
            create_graph=True
        )[0]
        d2xdz2_col = torch.autograd.grad(
            outputs=dxdz_col,
            inputs=z_col,
            grad_outputs=torch.ones_like(dxdz_col),
            create_graph=True
        )[0]

        residual = d2xdz2_col + 2.0 * xi_col * dxdz_col + x_pred_col
        loss_residual = torch.mean(residual ** 2)

        loss = loss_ic_val + loss_ic_deriv + loss_residual

        loss.backward()
        optimizer.step()

        # scheduler step setiap 1000 epoch (seperti kode Anda)
        if (epoch + 1) % 1000 == 0:
            scheduler.step()

        # -------- Early Stopping + Checkpoint --------
        improved = loss.item() + 1e-12 < best_loss
        if improved:
            best_loss = loss.item()
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": pinn.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "layers": layers,
                },
                ckpt_path,
            )
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 500 == 0 or improved:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch [{epoch+1}/{epochs}] "
                f"Loss: {loss.item():.6f} | IC_val: {loss_ic_val.item():.6f} | "
                f"IC_der: {loss_ic_deriv.item():.6f} | Res: {loss_residual.item():.6f} | "
                f"LR: {lr_now:.6f} | Best: {best_loss:.6f} | Pat:{epochs_no_improve}"
            )

        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best loss = {best_loss:.6f}")
            break

    # load best checkpoint
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        pinn.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded best checkpoint: epoch={ckpt['epoch']}, best_loss={ckpt['best_loss']:.6f}")

    return pinn

# =========================
# Evaluation: MSE table
# =========================
def evaluate_mse(pinn, xi_values=(0.1, 0.2, 0.3, 0.4), z_max=20.0, n_points=5000, save_csv="dho_mse_table.csv"):
    pinn.eval()
    results = []
    with torch.no_grad():
        for xi_val in xi_values:
            z = torch.linspace(0.0, z_max, n_points, device=device).view(-1, 1)
            xi = torch.full_like(z, float(xi_val))
            inp = torch.cat([z, xi], dim=1)
            x_pred = pinn(inp)
            x_true = analytical_solution(z, xi)
            mse = torch.mean((x_pred - x_true)**2).item()
            results.append((xi_val, mse))

    # print table
    print("\n=== MSE over z∈[0,{:.1f}] ===".format(z_max))
    for xi_val, mse in results:
        print(f"xi = {xi_val:.1f}  |  MSE = {mse:.3e}")

    # save CSV
    try:
        with open(save_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["xi", "MSE"])
            for xi_val, mse in results:
                writer.writerow([xi_val, mse])
        print(f"MSE table saved to: {save_csv}")
    except Exception as e:
        print(f"Warning: failed to save CSV ({e})")

    return results

# =========================
# Plotting
# =========================
def plot_results(pinn, save_path="damped_oscillator_results.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    axes = axes.flatten()
    xi_values = [0.1, 0.2, 0.3, 0.4]

    for i, xi_val in enumerate(xi_values):
        z_test = torch.linspace(0, 20, 500, device=device).view(-1, 1)
        xi_test = torch.full_like(z_test, xi_val)
        test_input = torch.cat([z_test, xi_test], dim=1)
        with torch.no_grad():
            x_pred_test = pinn(test_input).cpu().numpy()
        x_analytical = analytical_solution(z_test, xi_test).cpu().numpy()
        z_np = z_test.cpu().numpy()

        ax = axes[i]
        ax.plot(z_np, x_analytical, label="Analytical", linewidth=2)
        ax.plot(z_np, x_pred_test, linestyle="--", label="PINN", linewidth=2)
        ax.set_title(rf"Damped Harmonic Oscillator ($\xi$ = {xi_val})", fontsize=12, pad=8)
        ax.set_xlabel("Time (z)", labelpad=6)
        ax.set_ylabel("Displacement (x)", labelpad=6)
        ax.tick_params(axis="both", which="major", labelsize=9)
        ax.grid(True)
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)

    plt.savefig(save_path, dpi=160)
    plt.show()
    print(f"Plot saved to: {save_path}")

# =========================
# Main
# =========================
if __name__ == "__main__":
    pinn = train_pinn(
        epochs=20000,
        n_ic=100,
        n_col=2000,
        lr=1e-3,
        gamma=0.99,
        patience=1500,
        ckpt_path="pinn_dho_best.pt",
    )
    pinn.eval()
    plot_results(pinn)
    # Tambahan: evaluasi MSE per xi dan simpan tabel
    evaluate_mse(pinn, xi_values=(0.1, 0.2, 0.3, 0.4), z_max=20.0, n_points=5000, save_csv="dho_mse_table.csv")
