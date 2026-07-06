# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# ==========================================================================
#  Rectified-flow U-Net for KHI field generation
#
#  Predicts the rectified-flow velocity v(x_t, t) for the full 4-channel
#  field (density, velocity_x, velocity_y, pressure).
#
#  Design notes
#  ------------
#  * DDPM-style residual blocks (GroupNorm + SiLU + 3x3 conv) with FiLM
#    time conditioning (per-channel scale & shift). FiLM is applied at full
#    strength -- time information is essential for a flow model.
#  * Channel widths grow with depth; the architecture is parameterised by a
#    list of widths so there is no fragile hand-tuned channel arithmetic.
# ==========================================================================

import jax
import jax.numpy as jnp
import equinox as eqx

# ==========================================================================
#  constants
# ==========================================================================

INPUT_CHANNELS = 4          # density, velocity_x, velocity_y, pressure
WIDTHS = (64, 128, 256, 384)  # encoder stage widths (resolutions 256,128,64,32)
BOTTLENECK = 512            # bottleneck width (resolution 16)
FOURIER_DIM = 32
TIME_CHANNELS = FOURIER_DIM * 2   # dim of the raw fourier embedding
TIME_EMB = 256              # dim of the processed time embedding fed to FiLM
GROUPS = 8
MAX_PERIOD = 1000.0


# ==========================================================================
#  Fourier time embedding
# ==========================================================================

def fourier_embedding(t):
    """DDPM-style sinusoidal embedding of a scalar time t in [0, 1]."""
    d = 2
    freqs = jnp.array([MAX_PERIOD ** (-k / d) for k in range(FOURIER_DIM)])
    angles = freqs * t
    return jnp.concatenate([jnp.sin(angles), jnp.cos(angles)])


class TimeMLP(eqx.Module):
    """Learns which fourier frequencies matter for conditioning."""
    l1: eqx.nn.Linear
    l2: eqx.nn.Linear

    def __init__(self, in_dim, hidden_dim, out_dim, key):
        key1, key2 = jax.random.split(key)
        self.l1 = eqx.nn.Linear(in_dim, hidden_dim, key=key1)
        self.l2 = eqx.nn.Linear(hidden_dim, out_dim, key=key2)

    def __call__(self, t):
        h = jax.nn.silu(self.l1(t))
        return self.l2(h)


# ==========================================================================
#  Residual block with FiLM time conditioning
# ==========================================================================

class ResBlock(eqx.Module):
    norm1: eqx.nn.GroupNorm
    conv1: eqx.nn.Conv2d
    norm2: eqx.nn.GroupNorm
    conv2: eqx.nn.Conv2d
    film: eqx.nn.Linear
    skip: eqx.Module   # Conv2d (1x1) or Identity

    def __init__(self, in_ch, out_ch, t_dim, key):
        k1, k2, k3, k4 = jax.random.split(key, 4)
        self.norm1 = eqx.nn.GroupNorm(GROUPS, in_ch)
        self.conv1 = eqx.nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, key=k1)
        self.norm2 = eqx.nn.GroupNorm(GROUPS, out_ch)
        self.conv2 = eqx.nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, key=k2)
        # FiLM produces a per-channel (scale, shift) pair
        self.film = eqx.nn.Linear(t_dim, 2 * out_ch, key=k3)
        if in_ch == out_ch:
            self.skip = eqx.nn.Identity()
        else:
            self.skip = eqx.nn.Conv2d(in_ch, out_ch, kernel_size=1, key=k4)

    def __call__(self, x, t_emb):
        h = self.conv1(jax.nn.silu(self.norm1(x)))

        scale, shift = jnp.split(self.film(t_emb), 2, axis=-1)
        h = self.norm2(h)
        h = h * (1.0 + scale[:, None, None]) + shift[:, None, None]
        h = self.conv2(jax.nn.silu(h))

        return h + self.skip(x)


class Downsample(eqx.Module):
    conv: eqx.nn.Conv2d

    def __init__(self, ch, key):
        self.conv = eqx.nn.Conv2d(ch, ch, kernel_size=4, stride=2, padding=1, key=key)

    def __call__(self, x):
        return self.conv(x)


class Upsample(eqx.Module):
    conv: eqx.nn.Conv2d

    def __init__(self, ch, key):
        self.conv = eqx.nn.Conv2d(ch, ch, kernel_size=3, padding=1, key=key)

    def __call__(self, x):
        x = jax.image.resize(
            x, (x.shape[0], x.shape[1] * 2, x.shape[2] * 2), method="bilinear"
        )
        return self.conv(x)


# ==========================================================================
#  U-Net
# ==========================================================================

class UNet(eqx.Module):
    in_conv: eqx.nn.Conv2d
    down_blocks: list
    downsamples: list
    mid1: ResBlock
    mid2: ResBlock
    up_blocks: list
    upsamples: list
    t_mlp: TimeMLP
    out_norm: eqx.nn.GroupNorm
    out_conv: eqx.nn.Conv2d

    def __init__(self, key):
        keys = iter(jax.random.split(key, 64))

        widths = list(WIDTHS)
        self.in_conv = eqx.nn.Conv2d(INPUT_CHANNELS, widths[0], 3, padding=1, key=next(keys))
        self.t_mlp = TimeMLP(TIME_CHANNELS, TIME_EMB, TIME_EMB, key=next(keys))

        # ---- encoder ----
        self.down_blocks = []
        self.downsamples = []
        for i in range(len(widths)):
            in_ch = widths[i - 1] if i > 0 else widths[0]
            self.down_blocks.append(ResBlock(in_ch, widths[i], TIME_EMB, next(keys)))
            self.downsamples.append(Downsample(widths[i], next(keys)))

        # ---- bottleneck ----
        self.mid1 = ResBlock(widths[-1], BOTTLENECK, TIME_EMB, next(keys))
        self.mid2 = ResBlock(BOTTLENECK, BOTTLENECK, TIME_EMB, next(keys))

        # ---- decoder (mirror) ----
        self.up_blocks = []
        self.upsamples = []
        prev = BOTTLENECK
        for i in reversed(range(len(widths))):
            self.upsamples.append(Upsample(prev, next(keys)))
            # after upsample we concat the encoder skip of width widths[i]
            self.up_blocks.append(ResBlock(prev + widths[i], widths[i], TIME_EMB, next(keys)))
            prev = widths[i]

        self.out_norm = eqx.nn.GroupNorm(GROUPS, widths[0])
        self.out_conv = eqx.nn.Conv2d(widths[0], INPUT_CHANNELS, 1, key=next(keys))

    def __call__(self, x, t):
        t_emb = self.t_mlp(fourier_embedding(t))

        h = self.in_conv(x)

        skips = []
        for block, down in zip(self.down_blocks, self.downsamples):
            h = block(h, t_emb)
            skips.append(h)
            h = down(h)

        h = self.mid1(h, t_emb)
        h = self.mid2(h, t_emb)

        for up, block, skip in zip(self.upsamples, self.up_blocks, reversed(skips)):
            h = up(h)
            h = jnp.concatenate([h, skip], axis=0)
            h = block(h, t_emb)

        h = jax.nn.silu(self.out_norm(h))
        return self.out_conv(h)


# ==========================================================================
#  saving, loading and creation of models
# ==========================================================================

def create_model(key):
    return UNet(key)


def save_model(model: UNet, filepath: str):
    eqx.tree_serialise_leaves(filepath, model)


def load_model(filepath: str, key):
    model = UNet(key)
    return eqx.tree_deserialise_leaves(filepath, model)
