# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# ==========================================================================
#  Rectified-flow training for the 4-channel KHI field sampler
#
#  Learns v(x_t, t) = x1 - x0 where x0 ~ N(0, I), x1 is a (normalised) KHI
#  final state, and x_t = (1 - t) x0 + t x1.  Sampling then integrates this
#  velocity field from noise (t=0) to data (t=1).
# ==========================================================================

import argparse
import glob
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import matplotlib.pyplot as plt

from unet_models import unet_flow_film


# ==========================================================================
#  rectified-flow batch sampling
# ==========================================================================

def sample_batch(key, data, batch_size):
    """Draw a batch of (x_t, t, v_target) rectified-flow training triples."""
    keys = jax.random.split(key, batch_size)

    def single_sample(k):
        key1, key2, key3 = jax.random.split(k, 3)
        idx = jax.random.randint(key1, (), 0, len(data))
        x1 = jax.lax.dynamic_index_in_dim(data, idx, axis=0, keepdims=False)  # real KHI
        x0 = jax.random.normal(key2, x1.shape)                                # noise
        t = jax.random.uniform(key3, ())
        x_t = (1 - t) * x0 + t * x1
        v_target = x1 - x0                                                    # rectified flow
        return x_t, t, v_target

    return jax.vmap(single_sample)(keys)


def loss_function(model, x_t, t, v_target):
    v_pred = jax.vmap(model)(x_t, t)
    return jnp.mean((v_pred - v_target) ** 2)


@eqx.filter_jit
def train_step(model, ema_model, opt_state, x_t, t, v_target, optimizer, ema_decay):
    loss, grads = eqx.filter_value_and_grad(loss_function)(model, x_t, t, v_target)
    updates, opt_state = optimizer.update(grads, opt_state, eqx.filter(model, eqx.is_array))
    model = eqx.apply_updates(model, updates)

    # exponential moving average of the weights (used for sampling)
    ema_model = jax.tree_util.tree_map(
        lambda e, m: ema_decay * e + (1.0 - ema_decay) * m if eqx.is_array(e) else e,
        ema_model, model,
    )
    return model, ema_model, opt_state, loss


# ==========================================================================
#  data loading + normalisation
# ==========================================================================

def load_data(data_dir, max_samples=None):
    files = sorted(glob.glob(os.path.join(data_dir, "final_state_*.npy")))
    if max_samples is not None:
        files = files[:max_samples]
    if not files:
        raise FileNotFoundError(f"No data found in {data_dir}/final_state_*.npy")

    data = np.stack([np.load(f) for f in files]).astype(np.float32)  # (N, 4, 256, 256)

    # per-channel standardisation
    mean = data.mean(axis=(0, 2, 3), keepdims=True)  # (1, 4, 1, 1)
    std = data.std(axis=(0, 2, 3), keepdims=True)
    data = (data - mean) / std

    return jnp.asarray(data), mean.squeeze(), std.squeeze()  # stats shape (4,)


# ==========================================================================
#  main
# ==========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--ckpt-dir", type=str, default="unet_checkpoints")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)

    # ---- data ----
    data, mean, std = load_data(args.data_dir, args.max_samples)
    n_val = max(1, int(len(data) * args.val_frac))
    train_data, val_data = data[n_val:], data[:n_val]
    print(f"data: {data.shape}, train {len(train_data)}, val {len(val_data)}")
    print("per-channel mean:", np.round(np.array(mean), 4))
    print("per-channel std :", np.round(np.array(std), 4))

    # save normalisation stats for sampling
    np.savez(os.path.join(args.ckpt_dir, "norm_stats.npz"),
             mean=np.array(mean), std=np.array(std))

    # ---- model / optimiser ----
    key = jax.random.key(args.seed)
    key, mkey = jax.random.split(key)
    model = unet_flow_film.create_model(mkey)
    ema_model = model
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array)))
    print(f"model params: {n_params / 1e6:.2f} M")

    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(args.lr, weight_decay=1e-4),
    )
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    # ---- fixed validation batch ----
    x_val, t_val, v_val = sample_batch(jax.random.key(12345), val_data, batch_size=64)

    # ---- training loop ----
    loss_history, val_history, val_steps = [], [], []
    checkpoint_steps = {args.steps // 4, args.steps // 2, 3 * args.steps // 4, args.steps}

    key_train = jax.random.key(args.seed + 1)
    for step in range(1, args.steps + 1):
        key_train, subkey = jax.random.split(key_train)
        x_t, t, v_target = sample_batch(subkey, train_data, args.batch_size)
        model, ema_model, opt_state, loss = train_step(
            model, ema_model, opt_state, x_t, t, v_target, optimizer, args.ema_decay
        )
        loss_history.append(float(loss))

        if step % 500 == 0:
            val_mse = float(loss_function(ema_model, x_val, t_val, v_val))
            val_history.append(val_mse)
            val_steps.append(step)
            print(f"step {step:6d} | train {float(loss):.4f} | val(ema) {val_mse:.4f}",
                  flush=True)

        if step in checkpoint_steps:
            unet_flow_film.save_model(model, os.path.join(args.ckpt_dir, f"unet_{step}.eqx"))
            unet_flow_film.save_model(ema_model, os.path.join(args.ckpt_dir, f"unet_ema_{step}.eqx"))

    # ---- final checkpoints ----
    unet_flow_film.save_model(model, os.path.join(args.ckpt_dir, "unet_final.eqx"))
    unet_flow_film.save_model(ema_model, os.path.join(args.ckpt_dir, "unet_ema_final.eqx"))
    print("saved final checkpoints")

    # ---- plots ----
    plt.figure()
    plt.plot(loss_history, alpha=0.4, label="train (per step)")
    # smoothed
    if len(loss_history) > 100:
        w = 100
        smooth = np.convolve(loss_history, np.ones(w) / w, mode="valid")
        plt.plot(np.arange(w - 1, len(loss_history)), smooth, label="train (smoothed)")
    plt.plot(val_steps, val_history, "o-", label="val (ema)")
    plt.yscale("log")
    plt.xlabel("step")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.title("Rectified-flow training")
    plt.savefig("loss_history.png", dpi=120)
    print("wrote loss_history.png")


if __name__ == "__main__":
    main()
