# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# ==========================================================================
#  Sampling from the trained rectified-flow KHI model
#
#  Integrates the learned velocity field from Gaussian noise (t=0) to a KHI
#  final state (t=1) with a Heun (RK2) integrator, denormalises with the
#  saved per-channel statistics, and writes the full 4-channel field.
# ==========================================================================

import argparse
import glob
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from unet_models import unet_flow_film

CHANNELS = ["density", "velocity_x", "velocity_y", "pressure"]


# ==========================================================================
#  Heun (RK2) rectified-flow sampler
# ==========================================================================

def sample(model, x, steps=100):
    dt = 1.0 / steps
    for i in range(steps):
        t = i / steps
        v1 = model(x, t)
        x_euler = x + dt * v1
        t_next = min(t + dt, 1.0)
        v2 = model(x_euler, t_next)
        x = x + dt * 0.5 * (v1 + v2)
    return x


def batched_sample(model, noises, steps=100):
    return np.asarray(jax.vmap(lambda z: sample(model, z, steps))(noises))


# ==========================================================================
#  main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="unet_checkpoints/unet_ema_final.eqx")
    parser.add_argument("--ckpt-dir", type=str, default="unet_checkpoints")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--out-dir", type=str, default="unet_generations")
    parser.add_argument("--seed", type=int, default=999)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- load model + normalisation stats ----
    key = jax.random.key(0)
    model = unet_flow_film.load_model(args.ckpt, key)

    stats = np.load(os.path.join(args.ckpt_dir, "norm_stats.npz"))
    mean = stats["mean"].reshape(4, 1, 1)  # (4,1,1)
    std = stats["std"].reshape(4, 1, 1)

    # ---- generate ----
    noises = jax.random.normal(jax.random.key(args.seed), (args.num_samples, 4, 256, 256))
    gen_norm = batched_sample(model, noises, args.steps)          # (N, 4, 256, 256)
    gen = gen_norm * std[None] + mean[None]                       # denormalise

    np.save(os.path.join(args.out_dir, "generations.npy"), gen)
    print(f"generated {gen.shape} -> {args.out_dir}/generations.npy")

    # ---- reference (real) samples for side-by-side comparison ----
    real_files = sorted(glob.glob(os.path.join(args.data_dir, "final_state_*.npy")))
    real = np.stack([np.load(f) for f in real_files[:args.num_samples]]) if real_files else None

    # ---- per-channel stats sanity check ----
    print("\nchannel        gen[min,mean,max]            real[min,mean,max]")
    for c, name in enumerate(CHANNELS):
        g = gen[:, c]
        line = f"{name:12s} [{g.min():7.3f},{g.mean():7.3f},{g.max():7.3f}]"
        if real is not None:
            r = real[:, c]
            line += f"   [{r.min():7.3f},{r.mean():7.3f},{r.max():7.3f}]"
        print(line)

    # ---- visualise first few generated samples (all 4 channels) ----
    n_show = min(4, args.num_samples)
    fig, axes = plt.subplots(n_show, 4, figsize=(16, 4 * n_show))
    axes = np.atleast_2d(axes)
    for i in range(n_show):
        for c, name in enumerate(CHANNELS):
            ax = axes[i, c]
            im = ax.imshow(gen[i, c].T, origin="lower", cmap="jet")
            plt.colorbar(im, ax=ax, fraction=0.046)
            if i == 0:
                ax.set_title(name)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Generated KHI fields")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "generated_fields.png"), dpi=110)
    print(f"wrote {args.out_dir}/generated_fields.png")

    # ---- generated vs real density comparison ----
    if real is not None:
        fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
        for i in range(n_show):
            axes[0, i].imshow(gen[i, 0].T, origin="lower", cmap="jet")
            axes[0, i].set_title(f"gen {i}"); axes[0, i].axis("off")
            axes[1, i].imshow(real[i, 0].T, origin="lower", cmap="jet")
            axes[1, i].set_title(f"real {i}"); axes[1, i].axis("off")
        axes[0, 0].set_ylabel("generated")
        fig.suptitle("Density: generated (top) vs real (bottom)")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "density_compare.png"), dpi=110)
        print(f"wrote {args.out_dir}/density_compare.png")

        # ---- 2x4 overview: true (top) vs synthetic (bottom) for all channels ----
        # shared colour scale per channel so the two rows are directly comparable
        fig, axes = plt.subplots(2, 4, figsize=(18, 9))
        for c, name in enumerate(CHANNELS):
            vmin = min(real[0, c].min(), gen[0, c].min())
            vmax = max(real[0, c].max(), gen[0, c].max())
            im = axes[0, c].imshow(real[0, c].T, origin="lower", cmap="jet",
                                   vmin=vmin, vmax=vmax)
            axes[1, c].imshow(gen[0, c].T, origin="lower", cmap="jet",
                              vmin=vmin, vmax=vmax)
            axes[0, c].set_title(name)
            plt.colorbar(im, ax=axes[:, c], fraction=0.046, location="bottom", pad=0.04)
            for r in (0, 1):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[0, 0].set_ylabel("true", fontsize=14)
        axes[1, 0].set_ylabel("synthetic", fontsize=14)
        fig.suptitle("KHI fields — true (top) vs synthetic (bottom)", fontsize=15)
        fig.savefig(os.path.join(args.out_dir, "overview_2x4.png"), dpi=120,
                    bbox_inches="tight")
        print(f"wrote {args.out_dir}/overview_2x4.png")


if __name__ == "__main__":
    main()
