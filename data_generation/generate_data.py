# ==== GPU / XLA memory configuration (must precede any JAX import) ====
import os as _os
_os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
_os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus=1)
# ruff: noqa: E402
# =======================

# ==========================================================================
#  KHI dataset generation
#
#  Generates final-state Kelvin-Helmholtz snapshots with astronomix and
#  stores them as (4, 256, 256) arrays (density, velocity_x, velocity_y,
#  pressure) into the ``data/`` folder (a symlink to a large partition).
#
#  Simulations are batched with jax.vmap for throughput.
# ==========================================================================

import argparse
import os
from timeit import default_timer as timer

import jax
import jax.numpy as jnp
import numpy as np
from jax.random import PRNGKey

from astronomix import (
    SimulationConfig,
    SimulationParams,
    construct_primitive_state,
    get_helper_data,
    get_registered_variables,
    time_integration,
)
from astronomix.option_classes.simulation_config import (
    DOUBLE_MINMOD,
    FINITE_VOLUME,
    FORWARDS,
    HYBRID_HLLC,
    PERIODIC_BOUNDARY,
    BoundarySettings,
    BoundarySettings1D,
    finalize_config,
)
from astronomix.variable_registry.registered_variables import StaticIntVector

# ==========================================================================
#  simulation configuration
# ==========================================================================

NUM_CELLS = 256
BOX_SIZE = 1.0

config = SimulationConfig(
    progress_bar=False,
    dimensionality=2,
    box_size=BOX_SIZE,
    num_cells=StaticIntVector(x=NUM_CELLS, y=NUM_CELLS),
    fixed_timestep=False,
    differentiation_mode=FORWARDS,
    num_timesteps=2000,
    boundary_settings=BoundarySettings(
        x=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
        y=BoundarySettings1D(PERIODIC_BOUNDARY, PERIODIC_BOUNDARY),
    ),
    limiter=DOUBLE_MINMOD,
    return_snapshots=False,
    riemann_solver=HYBRID_HLLC,
    solver_mode=FINITE_VOLUME,
)

helper_data = get_helper_data(config)
params = SimulationParams(t_end=2.0, C_cfl=0.4)
registered_variables = get_registered_variables(config)

x = jnp.linspace(0, BOX_SIZE, NUM_CELLS)
y = jnp.linspace(0, BOX_SIZE, NUM_CELLS)
X, Y = jnp.meshgrid(x, y, indexing="ij")


# ==========================================================================
#  KHI initial condition
# ==========================================================================

def random_khi_fourier_modes(
    key,
    amplitude=0.01,
    k_min=1,
    k_max=8,
    shear_layers=(0.25, 0.75),
    width=0.03,
    spectral_slope=1.0,
):
    """Random Fourier-mode y-velocity perturbation for KHI seeding."""
    modes = jnp.arange(k_min, k_max + 1)
    num_modes = modes.shape[0]

    key_amp, key_phase = jax.random.split(key)

    coeffs = jax.random.normal(key_amp, (num_modes,)) / modes**spectral_slope
    coeffs = coeffs / jnp.sqrt(jnp.sum(coeffs**2) + 1e-30)

    phases = jax.random.uniform(key_phase, (num_modes,), minval=0.0, maxval=2.0 * jnp.pi)

    fourier_sum = jnp.sum(
        coeffs[:, None, None]
        * jnp.sin(2.0 * jnp.pi * modes[:, None, None] * X[None, :, :] + phases[:, None, None]),
        axis=0,
    )

    envelope = jnp.zeros_like(Y)
    for y0 in shear_layers:
        envelope = envelope + jnp.exp(-0.5 * ((Y - y0) / width) ** 2)

    return amplitude * envelope * fourier_sum


def make_initial_state(key):
    """Construct a KHI primitive initial state with a random perturbation."""
    rho = jnp.ones_like(X)
    u_x = 0.5 * jnp.ones_like(X)

    mask = (Y > 0.25) & (Y < 0.75)
    u_x = jnp.where(mask, -0.5, u_x)
    rho = jnp.where(mask, 2.0, rho)

    u_y = random_khi_fourier_modes(key)
    p = jnp.ones((NUM_CELLS, NUM_CELLS)) * 2.5

    return construct_primitive_state(
        config=config,
        registered_variables=registered_variables,
        density=rho,
        velocity_x=u_x,
        velocity_y=u_y,
        gas_pressure=p,
    )


# finalize config against a representative state shape
_init0 = make_initial_state(PRNGKey(0))
config = finalize_config(config, _init0.shape)


@jax.jit
@jax.vmap
def simulate_batch(initial_states):
    return time_integration(initial_states, config, params, registered_variables)


# ==========================================================================
#  generation loop
# ==========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="data")
    parser.add_argument("--overwrite", action="store_true",
                        help="regenerate even if the output file already exists")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    master_key = PRNGKey(args.seed)
    n = args.num_samples
    bs = args.batch_size

    t_start = timer()
    done = 0
    for batch_start in range(0, n, bs):
        this_bs = min(bs, n - batch_start)
        indices = list(range(batch_start, batch_start + this_bs))

        # skip fully-completed batches unless overwriting
        paths = [os.path.join(args.out_dir, f"final_state_{i:05d}.npy") for i in indices]
        if not args.overwrite and all(os.path.exists(p) for p in paths):
            done += this_bs
            continue

        # deterministic per-sample keys (independent of batch size)
        keys = jax.vmap(lambda i: jax.random.fold_in(master_key, i))(jnp.array(indices))
        initial_states = jax.vmap(make_initial_state)(keys)

        final_states = simulate_batch(initial_states)
        final_states.block_until_ready()
        final_states = np.asarray(final_states, dtype=np.float32)

        if np.isnan(final_states).any():
            bad = np.isnan(final_states).any(axis=(1, 2, 3))
            print(f"⚠️  NaNs in batch {batch_start}: samples {np.array(indices)[bad]}")

        for path, arr in zip(paths, final_states):
            np.save(path, arr)  # shape (4, 256, 256)

        done += this_bs
        elapsed = timer() - t_start
        rate = done / elapsed
        eta = (n - done) / rate if rate > 0 else float("nan")
        print(f"  {done:5d}/{n} done | {rate:5.2f} sim/s | ETA {eta/60:5.1f} min", flush=True)

    print(f"✅ Generated {n} samples in {(timer() - t_start)/60:.1f} min -> {args.out_dir}/")


if __name__ == "__main__":
    main()
