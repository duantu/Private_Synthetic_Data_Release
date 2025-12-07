"""
Minimal test driver for spiky nonconvex optimization.
"""
"""
Minimal test driver for spiky nonconvex optimization (no pre-run).
Relies on the runner to compute and report initial metrics.
"""

"""
Minimal test driver for spiky nonconvex optimization (no pre-run).
Relies on the runner to compute and report initial metrics.
"""
import numpy as np
import math
import secrets
import argparse
# from spiky_nonconvex_with_escape import SpikyNonconvexCoordinateDescent
from spiky_nonconvex_newton_gd import SpikyNonconvexCoordinateDescent

# ---- SINGLE PLACE TO EDIT ----
CONFIG = {
    "num_runs": 10,

    "n": 5000,          # keep
    "k": 40,            # keep
    "epsilon_base": 3,  # base mechanism privacy ε
    "delta_base": 1e-6,     # ↓ from 1e-4 → ↑ λ (via log(1/δ))
    "beta": 0.2,        # keep (per-query failure prob for base mech)
    "eta": 0.05,

    "max_iterations": 5000,
    "seed": None,
    "tau": 1e-20,      # let more outer steps happen before stopping on tiny loss updates
    "upper_bound": math.pi,
    "lower_bound": -math.pi,
    "freq_low": 0.5,
    "freq_high": 2,
    "amp_low": 0.1,
    "amp_high": 1.0,
    "offset": 0.0,
}


def build_optimizer(n, epsilon_base, delta_base, beta, eta, tau, upper_bound, lower_bound,offset):
    return SpikyNonconvexCoordinateDescent(
        epsilon=epsilon_base,
        delta=delta_base,
        beta=beta,
        eta=eta,
        n=n,
        tau=tau,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        offset=offset,
    )


def test_spiky_nonconvex_queries(
    n=120, k=36, epsilon_base=3.0, delta_base=1e-6, beta=0.1, eta=0.01,
    max_iterations=5000, seed=None,
    tau=0.01, upper_bound=math.pi, lower_bound=-math.pi,
    freq_low=4.5, freq_high=7.5, amp_low=0.1, amp_high=1.0, offset=0.0
):
    # Seed selection
    if seed is None:
        seed = secrets.randbits(32)
    np.random.seed(seed)  # drives amplitudes/frequencies

    # Build optimizer
    opt = build_optimizer(
        n=n,
        epsilon_base=epsilon_base,
        delta_base=delta_base,
        beta=beta,
        eta=eta,
        tau=tau,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        offset=offset,
    )

    # Random queries (amplitudes & freqs)
    amplitudes_matrix = np.random.uniform(amp_low, amp_high, size=(k, n))
    frequencies_vector = np.random.uniform(freq_low, freq_high, size=k)
    weights = [1.0 / k] * k
    opt.set_queries_and_amplitudes(amplitudes_matrix, frequencies_vector, weights)

    # Generate data
    opt.generate_data()

    # Compute outputs once so DP noise is fixed; also allows computing initial loss
    opt.compute_query_outputs()

    # Now it's safe to print lambda-related info
    target = 0.5 + eta
    lam = float(opt.lambda_val)
    noise_scale = opt.rho * math.sqrt(2 * k * math.log(1 / delta_base)) / epsilon_base
    print(f"noise_scale={noise_scale:.6f}, lambda/2={lam/2:.6f}")

    # Initial loss BEFORE any optimization
    loss_before = float(
        (
            np.exp((opt.fake_output - opt.real_data_noisy_output) / lam - 1.0)
            + np.exp((opt.real_data_noisy_output - opt.fake_output) / lam - 1.0)
        ).sum()
    )

    # Run optimizer WITHOUT resampling noise
    res = opt.run_coordinate_descent(
        max_iterations=max_iterations,
        verbose=False,
        resample_noise=False,
    )

    # Post-run metrics
    iterations = res.get("iterations", res.get("num_iterations", 0))
    reached = res.get("target_reached", False)  # informational if you changed stopping rule
    final_satisfaction = float(opt.compute_weighted_satisfaction_ratio())
    initial_satisfaction = float(res.get("initial_weighted_satisfaction", float("nan")))

    # Final loss
    loss_after = float(
        res.get(
            "final_loss",
            (
                np.exp((opt.fake_output - opt.real_data_noisy_output) / lam - 1.0)
                + np.exp((opt.real_data_noisy_output - opt.fake_output) / lam - 1.0)
            ).sum(),
        )
    )
    loss_drop = loss_before - loss_after

    # Report
    print(
        f"loss_before={loss_before:.6f} | loss_after={loss_after:.6f} | "
        f"loss_drop={loss_drop:.6f}"
    )
    print(
        f"seed={seed} | n={n} k={k} eps_base={epsilon_base} delta_base={delta_base} "
        f"beta={beta} eta={eta} "
        f"| target={target:.3f} | initial_satisfaction={initial_satisfaction:.3f} "
        f"| iters={iterations} | reached={reached} | "
        f"final_satisfaction={final_satisfaction:.3f}"
    )

    return {
        "seed": seed,
        "n": n,
        "k": k,
        "epsilon_base": epsilon_base,
        "delta_base": delta_base,
        "beta": beta,
        "eta": eta,
        "target": target,
        "iterations": iterations,
        "reached": reached,
        "initial_satisfaction": initial_satisfaction,
        "final_satisfaction": final_satisfaction,
        "lambda": lam,
        "noise_scale": float(noise_scale),
        "loss_before": float(loss_before),
        "loss_after": float(loss_after),
        "loss_drop": float(loss_drop),
    }


def test_multiple_runs(num_runs=10, seed=None, **kwargs):
    """
    Run multiple seeds. If 'seed' is supplied, use it as a base offset.
    All other parameters are forwarded to test_spiky_nonconvex_queries via **kwargs.
    """
    base = secrets.randbits(32) if seed is None else int(seed)
    for i in range(num_runs):
        s = base + i if seed is None else (seed + i)
        np.random.seed(s)
        _ = test_spiky_nonconvex_queries(seed=s, **kwargs)
        print("")


def parse_args_from_cli():
    p = argparse.ArgumentParser(description="Spiky Nonconvex tests")
    # Keep CLI optional; defaults come from CONFIG
    p.add_argument("--num_runs", type=int)
    p.add_argument("--n", type=int)
    p.add_argument("--k", type=int)
    p.add_argument("--epsilon_base", type=float)
    p.add_argument("--delta_base", type=float)
    p.add_argument("--beta", type=float)
    p.add_argument("--eta", type=float)
    p.add_argument("--max_iterations", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--tau", type=float)
    p.add_argument("--upper_bound", type=float)
    p.add_argument("--lower_bound", type=float)
    p.add_argument("--freq_low", type=float)
    p.add_argument("--freq_high", type=float)
    p.add_argument("--amp_low", type=float)
    p.add_argument("--amp_high", type=float)
    p.add_argument("--offset", type=float)
    return {k: v for k, v in vars(p.parse_args()).items() if v is not None}


if __name__ == "__main__":
    # Merge CLI overrides onto CONFIG; edit CONFIG for single-spot changes
    overrides = parse_args_from_cli()
    cfg = {**CONFIG, **overrides}

    test_multiple_runs(
        num_runs=cfg["num_runs"],
        seed=cfg["seed"],
        n=cfg["n"],
        k=cfg["k"],
        epsilon_base=cfg["epsilon_base"],
        delta_base=cfg["delta_base"],
        beta=cfg["beta"],
        eta=cfg["eta"],
        max_iterations=cfg["max_iterations"],
        tau=cfg["tau"],
        upper_bound=cfg["upper_bound"],
        lower_bound=cfg["lower_bound"],
        freq_low=cfg["freq_low"],
        freq_high=cfg["freq_high"],
        amp_low=cfg["amp_low"],
        amp_high=cfg["amp_high"],
        offset=cfg["offset"],
    )
