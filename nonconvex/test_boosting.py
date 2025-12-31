#!/usr/bin/env python3
"""
Test harness for the updated spiky boosting algorithm.

- Booster reads λ (lambda_val) and ρ (sensitivity) from the base optimizer
  each round, and computes μ from the theorem using the SAME ρ and your η as edge γ.
- Uses the ENTIRE query set each round and early-stops based on
  the per-query median-of-answers condition.
- Prints how many iterations the base synopsis generator ran each round.
"""

import math
from typing import Dict, List
import numpy as np

from spiky_boosting import (
    SpikyBoostingAlgorithm,
    compute_epsilon_sample_from_theorem,
    BaseSynopsisFailed,
)


from test_spiky_nonconvex_no_warmup import CONFIG as BASE_CONFIG


# ---------------------------------------------------------------------
# CONFIG: inherit everything from the non-warmup test, then add
#         boosting-specific knobs.
# ---------------------------------------------------------------------
CONFIG: Dict = dict(BASE_CONFIG)

# Number of boosting runs (can also just inherit BASE_CONFIG["num_runs"] if you prefer)
CONFIG["num_runs"] = 30

# Booster-only knobs (not used by the non-warmup driver)
CONFIG.setdefault("mu_constant", 1.0)          # C in the μ theorem (multiplicative scaling)
CONFIG.setdefault("min_rounds", 1)             # require at least this many boosting rounds
CONFIG.setdefault("stop_threshold_abs", None)  # optional absolute stopping threshold

# Theorem's T (separate from T_cap / max_iterations)
CONFIG.setdefault("theorem_T", 8)              # T used in μ theorem. This is an arbitrary number decided based on observation

# --- Booster's own ε_sample and δ_sample (PURELY boosting-level) ---
# These DO NOT default to the base ε, δ. Think of them as separate.
CONFIG.setdefault("epsilon_sample", 1)       # sample-level ε for the theorem
CONFIG.setdefault("delta_sample", 1e-1)        # sample-level δ for the theorem
CONFIG.setdefault("mu", 0.005)                   # desired μ (excess over λ)
CONFIG.setdefault("max_base_attempts", 30)



# Just in case you ever remove any of these from BASE_CONFIG in the future,
# keep safe defaults here.
CONFIG["max_iterations"] = int(math.ceil(math.log(CONFIG["k"]) / (CONFIG["eta"] ** 2))) # max number of boosting rounds needed, by Theorem 6.1
CONFIG.setdefault("tau", 1e-10)
CONFIG.setdefault("upper_bound", math.pi)
CONFIG.setdefault("lower_bound", -math.pi)
CONFIG.setdefault("freq_low", 1.2)
CONFIG.setdefault("freq_high", 2)
CONFIG.setdefault("amp_low", 0.1)
CONFIG.setdefault("amp_high", 1.0)
CONFIG.setdefault("offset", 0)

# Mixture of "easy" and "hard" queries (by amplitude only)
CONFIG.setdefault("p_hard", 0.3)   # fraction of queries drawn from the harder amplitude range
CONFIG.setdefault("amp_mid", 0.5)  # split point between easy and hard amplitudes


def generate_queries(
    k: int,
    n: int,
    amp_low: float,
    amp_high: float,
    freq_low: float,
    freq_high: float,
    rng: np.random.Generator,
):
    """
    Generate k heterogeneous queries, all dense over all coordinates.

    - EASY queries (fraction 1 - p_hard):
        * amplitudes in [amp_low, amp_mid]
        * frequencies in [freq_low, freq_high]        (low/medium frequency)
    - HARD queries (fraction p_hard):
        * amplitudes in [amp_mid, amp_high]
        * frequencies in [3*freq_high, 6*freq_high]   (high frequency)

    trig_flags:
        - alternate sin/cos within each group, then shuffle.
    """

    # Mixture controls from CONFIG
    p_hard = CONFIG.get("p_hard", 0.2)   # fraction of hard queries
    amp_mid = CONFIG.get("amp_mid", 0.5) # split point between easy and hard amplitudes

    # Clamp amp_mid into [amp_low, amp_high]
    amp_mid = max(amp_low, min(amp_mid, amp_high))

    k_hard = int(round(p_hard * k))
    k_easy = k - k_hard

    # Allocate amplitudes
    amplitudes_matrix = np.zeros((k, n), dtype=float)

    # ---------------- EASY QUERIES ----------------
    if k_easy > 0:
        amplitudes_matrix[:k_easy, :] = rng.uniform(
            amp_low, amp_mid, size=(k_easy, n)
        )
        freqs_easy = rng.uniform(freq_low, freq_high, size=k_easy)
    else:
        freqs_easy = np.array([], dtype=float)

    # ---------------- HARD QUERIES ----------------
    if k_hard > 0:
        amplitudes_matrix[k_easy:, :] = rng.uniform(
            amp_mid, amp_high, size=(k_hard, n)
        )

        # Hard queries get much higher frequency than the easy ones
        hard_freq_low = freq_high * 3.2
        hard_freq_high = freq_high * 4
        freqs_hard = rng.uniform(hard_freq_low, hard_freq_high, size=k_hard)
    else:
        freqs_hard = np.array([], dtype=float)

    frequencies_vector = np.concatenate([freqs_easy, freqs_hard])

    # ---------------- trig type: sin / cos ----------------
    trig_flags = np.zeros(k, dtype=bool)  # False → sin, True → cos

    # easy group: alternate sin/cos
    trig_flags[1:k_easy:2] = True

    # hard group: also alternate sin/cos within the group
    trig_flags[k_easy + 1 : k : 2] = True

    # ---------------- shuffle so easy/hard are mixed ----------------
    perm = rng.permutation(k)
    amplitudes_matrix = amplitudes_matrix[perm]
    frequencies_vector = frequencies_vector[perm]
    trig_flags = trig_flags[perm]

    return amplitudes_matrix, frequencies_vector, trig_flags





def generate_real_data(n: int, lower: float, upper: float, rng: np.random.Generator):
    """
    Generate real data in the same way the standalone driver does:
    real_X ~ N(0, 1). The lower/upper bounds are unused here, since
    the base optimizer's generate_data() also uses a Gaussian for real_X
    when no real_data is passed.
    """
    return rng.normal(loc=0.0, scale=1.0, size=n)


def run_single(cfg: Dict, rng: np.random.Generator, run_seed: int):

    n, k = cfg["n"], cfg["k"]

    # Booster-level theorem parameters
    epsilon_base = cfg["epsilon_base"]
    delta_base   = cfg["delta_base"]
    eta = cfg["eta"]
    delta_sample = cfg["delta_sample"]        # δ_sample
    mu = cfg["mu"]                            # desired μ
    beta = cfg["beta"]

    tau = cfg["tau"]
    lower, upper = cfg["lower_bound"], cfg["upper_bound"]

    # Sensitivity ρ: match SpikyNonconvexCoordinateDescent's formula
    rho = (upper ** 2 + 1.0) / n

    # Choose the theorem horizon T (you can tune theorem_T in CONFIG)
    T_theorem = int(cfg.get("theorem_T", 100))

    # ε_sample from the new theorem:
    epsilon_sample = compute_epsilon_sample_from_theorem(
        rho=rho,
        k=k,
        delta_sample=delta_sample,
        gamma_margin=eta,
        T=T_theorem,
        mu=mu,
    )

    # Use T_theorem as the boosting horizon T
    T_cap = T_theorem

    # Build synthetic query family and real data
    A, W, trig_flags = generate_queries(
        k,
        n,
        cfg["amp_low"],
        cfg["amp_high"],
        cfg["freq_low"],
        cfg["freq_high"],
        rng,
    )

    real_data = generate_real_data(n, lower, upper, rng)

    # Construct booster with fixed μ and T from the theorem
    booster = SpikyBoostingAlgorithm(
        k=k,
        lambda_param=None,          # ← pulled from base optimizer each round
        eta=eta,                    # ← edge γ
        rho=rho,                    # ← sensitivity (for logging)
        mu=mu,                      # ← PRE-DECIDED μ
        T=T_cap,                    # ← boosting horizon
        epsilon_base=cfg["epsilon_base"],
        delta_base=cfg["delta_base"],
        epsilon_sample=epsilon_sample,
        delta_sample=delta_sample,
        beta=beta,
        n=n,
        upper_bound=upper,
        lower_bound=lower,
        tau=tau,
        mu_constant=float(cfg.get("mu_constant", 1.0)),
        min_rounds=int(cfg.get("min_rounds", 1)),
        stop_threshold_abs=cfg.get("stop_threshold_abs", None),
        theorem_T=T_cap,            # for internal μ-theorem, if ever used
        offset=float(cfg.get("offset", 0.0)),
        max_base_attempts=int(cfg.get("max_base_attempts", 30)),
        repro_seed=run_seed,
    )

    booster.set_queries(A, W, trig_flags)

    print("=== Testing Spiky Boosting Algorithm (λ,ρ from base; fixed μ, T from μ-theorem) ===")
    print(
        f"n={n} | |Q|={k} | "
        f"ε_base={epsilon_base:.6f} δ_base={delta_base} ε_sample={epsilon_sample:.6f} δ_sample={delta_sample} β={beta} η={eta} | T={T_cap}"
    )
    print(f"OFFSET (additive constant) = {float(cfg.get('offset', 0.0)):.6f}")
    if cfg.get("stop_threshold_abs") is not None:
        print(f"Absolute stop threshold: {cfg['stop_threshold_abs']:.6f}")
    if int(cfg.get("min_rounds", 1)) > 1:
        print(f"Minimum rounds before stop: {int(cfg['min_rounds'])}")
    print(f"μ constant (C): {float(cfg.get('mu_constant', 1.0)):.3f}")
    print("Threshold scaling: thr = λ + μ")

    results = booster.run_boosting(real_data, verbose=True)

    # Show how many iterations the base optimizer ran each boosting round
    print("\n--- Base optimizer iterations by round ---")
    for s in results["all_synopses"]:
        print(f"t={s['iteration']}: {s.get('base_iterations', 'unknown')}")

    # Evaluate final answers against truth (using the final ensemble median)
    final = results["final_answers"]
    errors = []
    trig_flags_local = trig_flags  # length k
    for q in range(k):
        amps = A[q]
        w = W[q]
        if trig_flags_local[q]:
            trig = np.cos(w * real_data)
        else:
            trig = np.sin(w * real_data)
        real_ans = float(np.sum(real_data ** 2 + amps * trig) / n)
        errors.append(abs(float(final[q]) - real_ans))

    errors = np.array(errors, dtype=float)

    # Threshold used for reporting ONLY (not for boosting logic)
    lam = float(booster.lambda_param)
    mu = float(booster.mu)
    thr =  lam + mu
    if cfg.get("stop_threshold_abs") is not None:
        thr = min(thr, float(cfg["stop_threshold_abs"]))

    print("\n=== Final Statistics ===")
    print(f"Early stopped: {results['early_stopped']} after {results['iterations_run']} iteration(s)")
    print(f"λ (from base)={lam:.6f}, μ (fixed target)={mu:.6f}, threshold used={thr:.6f}")
    print(f"Share within threshold: {np.mean(errors <= thr):.3f}")

    return results, errors, lam, mu, thr


def test_boosting():
    seed = CONFIG["seed"]
    if seed is None:
        import time
        seed = int(time.time()) % (2**31 - 1)
    print(f"Global seed: {seed}")
    master_rng = np.random.default_rng(seed)

    all_max: List[float] = []
    all_thr: List[float] = []

    for run in range(CONFIG["num_runs"]):
        print(f"\n========== RUN {run + 1} / {CONFIG['num_runs']} ==========")
        run_seed = int(master_rng.integers(0, 2**63 - 1))
        print(f"[REPRO] global_seed={seed} run_index={run+1} run_seed={run_seed}")
        rng = np.random.default_rng(run_seed)
        try:
            _, errs, lam, mu, thr = run_single(CONFIG, rng, run_seed=run_seed)
        except BaseSynopsisFailed as e:
            print(f"[RUN FAILED] {e}")
            continue

        all_max.append(float(np.max(errs)))
        all_thr.append(float(thr))

    all_max = np.array(all_max, dtype=float)
    all_thr = np.array(all_thr, dtype=float)

    return True


if __name__ == "__main__":
    test_boosting()
