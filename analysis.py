# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# ==========================================================================
#  Physical analysis of generated vs. real KHI fields
#
#  Compares generated samples (from sample.py) against real simulation data
#  along physically meaningful statistics:
#    * per-channel value distributions (PDFs)
#    * density power spectrum P(k)
#    * kinetic energy spectrum E(k)
# ==========================================================================

import argparse
import glob
import math
import os

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

CHANNELS = ["density", "velocity_x", "velocity_y", "pressure"]

# ==========================================================================
#  Binning mode constants
# ==========================================================================

LOG_BINNING = 0
INTEGER_BINNING = 1
PHYSICAL_BINNING = 2


def _wavenumber_bins2d(Nx, Ny, binning=PHYSICAL_BINNING):
    """Shell-binning indices for spectral accumulation."""
    freq_x = jnp.fft.fftfreq(Nx, d=1.0 / Nx)
    freq_y = jnp.fft.fftfreq(Ny, d=1.0 / Ny)
    FX, FY = jnp.meshgrid(freq_x, freq_y, indexing="ij")
    m_mag = jnp.sqrt(FX**2 + FY**2)
    max_mode = math.sqrt((Nx // 2) ** 2 + (Ny // 2) ** 2)

    if binning == LOG_BINNING:
        k_min, k_max = 1.0, max_mode
        n_bins = max(Nx, Ny) // 4
        log_edges = jnp.logspace(jnp.log10(k_min * 0.5), jnp.log10(k_max + 0.5), n_bins + 1)
        k_idx = jnp.clip(jnp.digitize(m_mag.ravel(), log_edges) - 1, 0, n_bins - 1)
        k_centers = 2.0 * jnp.pi * jnp.sqrt(log_edges[:-1] * log_edges[1:])
        return k_idx, n_bins, k_centers
    elif binning == INTEGER_BINNING:
        k_idx = m_mag.astype(jnp.int32).ravel()
        n_bins = int(max_mode) + 2
        k_centers = (jnp.arange(n_bins) + 0.5) * 2.0 * jnp.pi
        return k_idx, n_bins, k_centers
    else:  # PHYSICAL_BINNING
        k_phys = 2.0 * jnp.pi * m_mag
        k_idx = k_phys.astype(jnp.int32).ravel()
        n_bins = int(2.0 * math.pi * max_mode) + 2
        k_centers = jnp.arange(n_bins) + 0.5
        return k_idx, n_bins, k_centers


# ==========================================================================
#  Spectra
# ==========================================================================

def vector_field_energy_spectrum(fx, fy, energy_coeff=1.0, binning=PHYSICAL_BINNING):
    """Energy spectrum E(k) of a vector field (fx, fy)."""
    Nx, Ny = fx.shape
    N_total = float(Nx * Ny)
    fx_hat = jnp.fft.fftn(fx)
    fy_hat = jnp.fft.fftn(fy)
    energy_fft = energy_coeff * (jnp.abs(fx_hat) ** 2 + jnp.abs(fy_hat) ** 2) / N_total**2
    k_idx, n_bins, k_centers = _wavenumber_bins2d(Nx, Ny, binning)
    Ek = jnp.zeros(n_bins).at[k_idx].add(energy_fft.ravel())
    return k_centers, Ek


def get_kinetic_energy_spectrum(vx, vy, rho, binning=PHYSICAL_BINNING):
    """Density-weighted kinetic energy spectrum (w = sqrt(rho) * u)."""
    rho_sqrt = jnp.sqrt(jnp.clip(rho, 1e-6))
    return vector_field_energy_spectrum(rho_sqrt * vx, rho_sqrt * vy,
                                        energy_coeff=0.5, binning=binning)


def density_power_spectrum(rho):
    """Radially-averaged power spectrum P(k) for a stack of density fields.

    rho.shape = (n_data, N, N) -> returns (k_centers, Pk_all[n_data, n_bins]).
    """
    rho = rho - rho.mean(axis=(-2, -1), keepdims=True)
    n_data, N, _ = rho.shape
    rho_hat = jnp.fft.fft2(rho, axes=(-2, -1))
    power = jnp.abs(rho_hat) ** 2
    k_idx, n_bins, k_centers = _wavenumber_bins2d(N, N, binning=PHYSICAL_BINNING)
    k_idx_flat = k_idx.ravel()
    counts = jnp.bincount(k_idx_flat, length=n_bins)
    Pk_all = []
    for j in range(n_data):
        Pk = jnp.bincount(k_idx_flat, weights=power[j].ravel(), length=n_bins)
        Pk = jnp.where(counts > 0, Pk / jnp.clip(counts, 1), 0.0)
        Pk_all.append(Pk)
    return k_centers, jnp.stack(Pk_all)


# ==========================================================================
#  Plots
# ==========================================================================

def plot_value_distributions(gen, real, out_path):
    """Histogram PDFs per channel: generated vs real."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    for c, name in enumerate(CHANNELS):
        ax = axes[c]
        g = np.asarray(gen[:, c]).ravel()
        r = np.asarray(real[:, c]).ravel()
        lo = min(g.min(), r.min())
        hi = max(g.max(), r.max())
        bins = np.linspace(lo, hi, 120)
        ax.hist(r, bins=bins, density=True, histtype="step", label="real", color="k")
        ax.hist(g, bins=bins, density=True, histtype="step", label="generated", color="r")
        ax.set_title(name)
        ax.set_xlabel("value")
        ax.set_yscale("log")
        ax.legend()
    axes[0].set_ylabel("PDF")
    fig.suptitle("Value distributions: generated vs real")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_spectra(gen, real, out_path):
    """Mean density power spectrum and kinetic energy spectrum."""
    kd, Pk_gen = density_power_spectrum(gen[:, 0])
    _, Pk_real = density_power_spectrum(real[:, 0])

    ke_gen = np.stack([np.asarray(get_kinetic_energy_spectrum(
        gen[i, 1], gen[i, 2], gen[i, 0])[1]) for i in range(len(gen))])
    ke_k, _ = get_kinetic_energy_spectrum(real[0, 1], real[0, 2], real[0, 0])
    ke_real = np.stack([np.asarray(get_kinetic_energy_spectrum(
        real[i, 1], real[i, 2], real[i, 0])[1]) for i in range(len(real))])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    def _band(ax, k, spec, color, label):
        m = np.asarray(spec).mean(0)
        ax.loglog(k, m, color=color, label=label)

    _band(ax1, kd, Pk_real, "k", "real")
    _band(ax1, kd, Pk_gen, "r", "generated")
    ax1.set_title("Density power spectrum P(k)")
    ax1.set_xlabel("k"); ax1.set_ylabel("P(k)"); ax1.legend()
    ax1.set_xlim(1, None)

    _band(ax2, ke_k, ke_real, "k", "real")
    _band(ax2, ke_k, ke_gen, "r", "generated")
    ax2.set_title("Kinetic energy spectrum E(k)")
    ax2.set_xlabel("k"); ax2.set_ylabel("E(k)"); ax2.legend()
    ax2.set_xlim(1, None)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    print(f"wrote {out_path}")


# ==========================================================================
#  main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen", type=str, default="unet_generations/generations.npy")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--out-dir", type=str, default="unet_generations")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    gen = np.load(args.gen).astype(np.float32)  # (N, 4, 256, 256)
    real_files = sorted(glob.glob(os.path.join(args.data_dir, "final_state_*.npy")))
    n = min(len(gen), len(real_files))
    real = np.stack([np.load(f) for f in real_files[:max(n, 64)]]).astype(np.float32)
    print(f"generated: {gen.shape}, real: {real.shape}")

    plot_value_distributions(gen, real, os.path.join(args.out_dir, "value_distributions.png"))
    plot_spectra(gen, real, os.path.join(args.out_dir, "spectra.png"))


if __name__ == "__main__":
    main()
