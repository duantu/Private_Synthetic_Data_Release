# # #!/usr/bin/env python3
# # """
# # Test harness for the updated spiky boosting algorithm.

# # - Booster reads λ (lambda_val) and ρ (sensitivity) from the base optimizer
# #   each round, and computes μ from the theorem using the SAME ρ and your η as edge γ.
# # - Uses the ENTIRE query set each round and early-stops based on
# #   the per-query median-of-answers condition.
# # - Prints how many iterations the base synopsis generator ran each round.
# # """

# # import math
# # from typing import Dict, List
# # import numpy as np

# # from spiky_boosting import SpikyBoostingAlgorithm
# # from test_spiky_nonconvex_no_warmup import CONFIG as BASE_CONFIG


# # # ---------------------------------------------------------------------
# # # CONFIG: inherit everything from the non-warmup test, then add
# # #         boosting-specific knobs.
# # # ---------------------------------------------------------------------
# # CONFIG: Dict = dict(BASE_CONFIG)

# # # Number of boosting runs (can also just inherit BASE_CONFIG["num_runs"] if you prefer)
# # CONFIG.setdefault("num_runs", 10)

# # # Booster-only knobs (not used by the non-warmup driver)
# # CONFIG.setdefault("mu_constant", 1.0)          # C in the μ theorem (multiplicative scaling)
# # CONFIG.setdefault("min_rounds", 1)             # require at least this many boosting rounds
# # CONFIG.setdefault("stop_threshold_abs", None)  # optional absolute stopping threshold

# # # --- Booster's own ε_sample and δ_sample (PURELY boosting-level) ---
# # # These DO NOT default to the base ε, δ. You MUST think of them as
# # # separate parameters from the theorem. Change these as you like.
# # CONFIG.setdefault("epsilon_sample", 0.5)       # placeholder; tune as needed
# # CONFIG.setdefault("delta_sample", 1e-2)        # placeholder; tune as needed

# # # Just in case you ever remove any of these from BASE_CONFIG in the future,
# # # keep safe defaults here. Right now your test_spiky_nonconvex_no_warmup
# # # already defines all of them, so these are effectively no-ops.
# # CONFIG.setdefault("max_iterations", 5000)
# # CONFIG.setdefault("tau", 1e-20)
# # CONFIG.setdefault("upper_bound", math.pi)
# # CONFIG.setdefault("lower_bound", -math.pi)
# # CONFIG.setdefault("freq_low", 4.5)
# # CONFIG.setdefault("freq_high", 7.5)
# # CONFIG.setdefault("amp_low", 0.1)
# # CONFIG.setdefault("amp_high", 1.0)


# # def generate_queries(
# #     k: int,
# #     n: int,
# #     amp_low: float,
# #     amp_high: float,
# #     freq_low: float,
# #     freq_high: float,
# #     rng: np.random.Generator,
# # ):
# #     amplitudes_matrix = rng.uniform(amp_low, amp_high, size=(k, n))
# #     frequencies_vector = rng.uniform(freq_low, freq_high, size=k)
# #     return amplitudes_matrix, frequencies_vector


# # def generate_real_data(n: int, lower: float, upper: float, rng: np.random.Generator):
# #     """
# #     Generate real data in the same way the standalone driver does:
# #     real_X ~ N(0, 1). The lower/upper bounds are unused here, since
# #     the base optimizer's generate_data() also uses a Gaussian for real_X
# #     when no real_data is passed.
# #     """
# #     return rng.normal(loc=0.0, scale=1.0, size=n)


# # def run_single(cfg: Dict, rng: np.random.Generator):
# #     n, k = cfg["n"], cfg["k"]

# #     # Booster-level theorem parameters
# #     eta = cfg["eta"]
# #     epsilon_sample = cfg["epsilon_sample"]    # sample-level ε for boosting theorem
# #     delta_sample = cfg["delta_sample"]        # sample-level δ for boosting theorem
# #     beta = cfg["beta"]

# #     tau, T_cap = cfg["tau"], cfg["max_iterations"]
# #     lower, upper = cfg["lower_bound"], cfg["upper_bound"]

# #     # Build synthetic query family and real data
# #     A, W = generate_queries(
# #         k,
# #         n,
# #         cfg["amp_low"],
# #         cfg["amp_high"],
# #         cfg["freq_low"],
# #         cfg["freq_high"],
# #         rng,
# #     )
# #     real_data = generate_real_data(n, lower, upper, rng)

# #     # Construct booster with λ, ρ, μ left as None → λ & ρ from base; μ from theorem with η as edge
# #     booster = SpikyBoostingAlgorithm(
# #         k=k,
# #         lambda_param=None,   # ← pulled from base optimizer
# #         eta=eta,             # ← used as edge γ in μ
# #         rho=None,            # ← pulled from base optimizer
# #         mu=None,             # ← computed from theorem
# #         T=T_cap,
# #         epsilon_sample=epsilon_sample,   # ← ε_sample for theorem (separate from base ε)
# #         delta=delta_sample,              # ← δ_sample for theorem (separate from base δ)
# #         beta=beta,
# #         n=n,
# #         upper_bound=upper,
# #         lower_bound=lower,
# #         tau=tau,
# #         mu_constant=float(cfg.get("mu_constant", 1.0)),
# #         min_rounds=int(cfg.get("min_rounds", 1)),
# #         stop_threshold_abs=cfg.get("stop_threshold_abs", None),
# #     )

# #     booster.set_queries(A, W)

# #     print("=== Testing Spiky Boosting Algorithm (λ,ρ from base; μ uses η as edge) ===")
# #     print(
# #         f"n={n} | |Q|={k} | "
# #         f"ε_sample={epsilon_sample} δ_sample={delta_sample} β={beta} η={eta} | T_cap={T_cap}"
# #     )
# #     if cfg.get("stop_threshold_abs") is not None:
# #         print(f"Absolute stop threshold: {cfg['stop_threshold_abs']:.6f}")
# #     if int(cfg.get("min_rounds", 1)) > 1:
# #         print(f"Minimum rounds before stop: {int(cfg['min_rounds'])}")
# #     print(
# #         "μ constant (C): "
# #         f"{float(cfg.get('mu_constant', 1.0)):.3f}"
# #     )

# #     results = booster.run_boosting(real_data, verbose=True)

# #     # Show how many iterations the base optimizer ran each boosting round
# #     print("\n--- Base optimizer iterations by round ---")
# #     for s in results["all_synopses"]:
# #         print(f"t={s['iteration']}: {s.get('base_iterations', 'unknown')}")

# #     # Evaluate final answers against truth
# #     final = results["final_answers"]
# #     errors = []
# #     for q in range(k):
# #         amps = A[q]
# #         w = W[q]
# #         real_ans = float(np.sum(real_data ** 2 + amps * np.sin(w * real_data)) / n)
# #         errors.append(abs(float(final[q]) - real_ans))
# #     errors = np.array(errors, dtype=float)

# #     # Threshold used (min with abs stop if provided)
# #     lam = float(booster.lambda_param)
# #     mu = float(booster.mu)
# #     # thr = lam + mu
# #     thr = 0.5*lam
# #     if cfg.get("stop_threshold_abs") is not None:
# #         thr = min(thr, float(cfg["stop_threshold_abs"]))

# #     print("\n=== Final Statistics ===")
# #     print(f"Early stopped: {results['early_stopped']} after {results['iterations_run']} iteration(s)")
# #     print(f"λ (from base)={lam:.6f}, μ (theorem w/ η edge)={mu:.6f}, threshold used={thr:.6f}")
# #     print(f"Share within threshold: {np.mean(errors <= thr):.3f}")

# #     return results, errors, lam, mu, thr


# # def test_boosting():
# #     seed = CONFIG["seed"]
# #     if seed is None:
# #         import time
# #         seed = int(time.time()) % (2**31 - 1)
# #     print(f"Global seed: {seed}")
# #     master_rng = np.random.default_rng(seed)

# #     all_max: List[float] = []
# #     all_thr: List[float] = []

# #     for run in range(CONFIG["num_runs"]):
# #         print(f"\n========== RUN {run + 1} / {CONFIG['num_runs']} ==========")
# #         rng = np.random.default_rng(master_rng.integers(0, 2**63 - 1))
# #         _, errs, lam, mu, thr = run_single(CONFIG, rng)
# #         all_max.append(float(np.max(errs)))
# #         all_thr.append(float(thr))

# #     all_max = np.array(all_max, dtype=float)
# #     all_thr = np.array(all_thr, dtype=float)

# #     return True


# # if __name__ == "__main__":
# #     test_boosting()

# #!/usr/bin/env python3
# """
# Test harness for the updated spiky boosting algorithm.

# - Booster reads λ (lambda_val) and ρ (sensitivity) from the base optimizer
#   each round, and computes μ from the theorem using the SAME ρ and your η as edge γ.
# - Uses the ENTIRE query set each round and early-stops based on
#   the per-query median-of-answers condition.
# - Prints how many iterations the base synopsis generator ran each round.
# """

# import math
# from typing import Dict, List
# import numpy as np

# from spiky_boosting import SpikyBoostingAlgorithm
# from test_spiky_nonconvex_no_warmup import CONFIG as BASE_CONFIG


# # ---------------------------------------------------------------------
# # CONFIG: inherit everything from the non-warmup test, then add
# #         boosting-specific knobs.
# # ---------------------------------------------------------------------
# CONFIG: Dict = dict(BASE_CONFIG)

# # Number of boosting runs
# CONFIG.setdefault("num_runs", 10)

# # Booster-only knobs (not used by the non-warmup driver)
# CONFIG.setdefault("mu_constant", 1.0)             # C in the μ theorem
# CONFIG.setdefault("min_rounds", 1)                # require at least this many boosting rounds
# CONFIG.setdefault("stop_threshold_abs", None)     # optional absolute stopping threshold

# # Booster's own ε_sample and δ_sample (for μ theorem)
# CONFIG.setdefault("epsilon_sample", 0.5)
# CONFIG.setdefault("delta_sample", 1e-2)

# # Threshold scaling: thr = threshold_scale_lambda * λ
# # e.g. 1.0 → thr = λ, 0.5 → thr = 0.5 λ
# CONFIG.setdefault("threshold_scale_lambda", 0.5)

# # Safety defaults (BASE_CONFIG already defines these, but keep them here)
# CONFIG.setdefault("max_iterations", 5000)
# CONFIG.setdefault("tau", 1e-20)
# CONFIG.setdefault("upper_bound", math.pi)
# CONFIG.setdefault("lower_bound", -math.pi)
# CONFIG.setdefault("freq_low", 4.5)
# CONFIG.setdefault("freq_high", 7.5)
# CONFIG.setdefault("amp_low", 0.1)
# CONFIG.setdefault("amp_high", 1.0)


# def generate_queries(
#     k: int,
#     n: int,
#     amp_low: float,
#     amp_high: float,
#     freq_low: float,
#     freq_high: float,
#     rng: np.random.Generator,
# ):
#     amplitudes_matrix = rng.uniform(amp_low, amp_high, size=(k, n))
#     frequencies_vector = rng.uniform(freq_low, freq_high, size=k)
#     return amplitudes_matrix, frequencies_vector


# def generate_real_data(n: int, lower: float, upper: float, rng: np.random.Generator):
#     """
#     Generate real data in the same way the standalone driver does:
#     real_X ~ N(0, 1). The lower/upper bounds are unused here, since
#     the base optimizer's generate_data() also uses a Gaussian for real_X
#     when no real_data is passed.
#     """
#     return rng.normal(loc=0.0, scale=1.0, size=n)


# def run_single(cfg: Dict, rng: np.random.Generator):
#     n, k = cfg["n"], cfg["k"]

#     # Booster-level theorem parameters
#     eta = cfg["eta"]
#     epsilon_sample = cfg["epsilon_sample"]    # ε_sample for theorem
#     delta_sample = cfg["delta_sample"]        # δ_sample for theorem
#     beta = cfg["beta"]

#     tau, T_cap = cfg["tau"], cfg["max_iterations"]
#     lower, upper = cfg["lower_bound"], cfg["upper_bound"]

#     threshold_scale_lambda = float(cfg.get("threshold_scale_lambda", 1.0))

#     # Build synthetic query family and real data
#     A, W = generate_queries(
#         k,
#         n,
#         cfg["amp_low"],
#         cfg["amp_high"],
#         cfg["freq_low"],
#         cfg["freq_high"],
#         rng,
#     )
#     real_data = generate_real_data(n, lower, upper, rng)

#     # Construct booster with λ, ρ, μ left as None → λ & ρ from base; μ from theorem with η as edge
#     booster = SpikyBoostingAlgorithm(
#         k=k,
#         lambda_param=None,   # pulled from base optimizer
#         eta=eta,             # used as edge γ in μ
#         rho=None,            # pulled from base optimizer
#         mu=None,             # computed from theorem
#         T=T_cap,
#         epsilon_sample=epsilon_sample,
#         delta=delta_sample,
#         beta=beta,
#         n=n,
#         upper_bound=upper,
#         lower_bound=lower,
#         tau=tau,
#         mu_constant=float(cfg.get("mu_constant", 1.0)),
#         min_rounds=int(cfg.get("min_rounds", 1)),
#         stop_threshold_abs=cfg.get("stop_threshold_abs", None),
#         threshold_scale_lambda=threshold_scale_lambda,
#     )

#     booster.set_queries(A, W)

#     print("=== Testing Spiky Boosting Algorithm (λ,ρ from base; μ uses η as edge) ===")
#     print(
#         f"n={n} | |Q|={k} | "
#         f"ε_sample={epsilon_sample} δ_sample={delta_sample} β={beta} η={eta} | T_cap={T_cap}"
#     )
#     if cfg.get("stop_threshold_abs") is not None:
#         print(f"Absolute stop threshold: {cfg['stop_threshold_abs']:.6f}")
#     if int(cfg.get("min_rounds", 1)) > 1:
#         print(f"Minimum rounds before stop: {int(cfg['min_rounds'])}")
#     print(
#         "μ constant (C): "
#         f"{float(cfg.get('mu_constant', 1.0)):.3f}"
#     )
#     print(
#         f"Threshold scaling: thr = {threshold_scale_lambda:.3f} * λ"
#     )

#     results = booster.run_boosting(real_data, verbose=True)

#     # Show how many iterations the base optimizer ran each boosting round
#     print("\n--- Base optimizer iterations by round ---")
#     for s in results["all_synopses"]:
#         print(f"t={s['iteration']}: {s.get('base_iterations', 'unknown')}")

#     # Use the booster’s own final stats: median ensemble + same threshold
#     lam = float(results["lambda"])
#     mu = float(results["mu"])
#     thr = float(results["threshold"])
#     errors = np.array(results["final_errors"], dtype=float)

#     print("\n=== Final Statistics ===")
#     print(f"Early stopped: {results['early_stopped']} after {results['iterations_run']} iteration(s)")
#     print(f"λ (from base)={lam:.6f}, μ (theorem w/ η edge)={mu:.6f}, threshold used={thr:.6f}")
#     print(f"Share within threshold: {np.mean(errors <= thr):.3f}")

#     return results, errors, lam, mu, thr


# def test_boosting():
#     seed = CONFIG["seed"]
#     if seed is None:
#         import time
#         seed = int(time.time()) % (2**31 - 1)
#     print(f"Global seed: {seed}")
#     master_rng = np.random.default_rng(seed)

#     all_max: List[float] = []
#     all_thr: List[float] = []

#     for run in range(CONFIG["num_runs"]):
#         print(f"\n========== RUN {run + 1} / {CONFIG['num_runs']} ==========")
#         rng = np.random.default_rng(master_rng.integers(0, 2**63 - 1))
#         _, errs, lam, mu, thr = run_single(CONFIG, rng)
#         all_max.append(float(np.max(errs)))
#         all_thr.append(float(thr))

#     all_max = np.array(all_max, dtype=float)
#     all_thr = np.array(all_thr, dtype=float)

#     # You can add summary stats here if you want
#     # e.g. print(f"Avg max error: {np.mean(all_max):.4f}")

#     return True


# if __name__ == "__main__":
#     test_boosting()

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

from spiky_boosting import SpikyBoostingAlgorithm
from test_spiky_nonconvex_no_warmup import CONFIG as BASE_CONFIG


# ---------------------------------------------------------------------
# CONFIG: inherit everything from the non-warmup test, then add
#         boosting-specific knobs.
# ---------------------------------------------------------------------
CONFIG: Dict = dict(BASE_CONFIG)

# Number of boosting runs (can also just inherit BASE_CONFIG["num_runs"] if you prefer)
CONFIG.setdefault("num_runs", 10)

# Booster-only knobs (not used by the non-warmup driver)
CONFIG.setdefault("mu_constant", 1.0)          # C in the μ theorem (multiplicative scaling)
CONFIG.setdefault("min_rounds", 1)             # require at least this many boosting rounds
CONFIG.setdefault("stop_threshold_abs", None)  # optional absolute stopping threshold

# Theorem's T (separate from T_cap / max_iterations)
CONFIG.setdefault("theorem_T", 5)              # T used in μ theorem

# --- Booster's own ε_sample and δ_sample (PURELY boosting-level) ---
# These DO NOT default to the base ε, δ. Think of them as separate.
CONFIG.setdefault("epsilon_sample", 0.5)       # sample-level ε for the theorem
CONFIG.setdefault("delta_sample", 1e-2)        # sample-level δ for the theorem

# Just in case you ever remove any of these from BASE_CONFIG in the future,
# keep safe defaults here.
CONFIG.setdefault("max_iterations", 5000)
CONFIG.setdefault("tau", 1e-20)
CONFIG.setdefault("upper_bound", math.pi)
CONFIG.setdefault("lower_bound", -math.pi)
CONFIG.setdefault("freq_low", 4.5)
CONFIG.setdefault("freq_high", 7.5)
CONFIG.setdefault("amp_low", 0.1)
CONFIG.setdefault("amp_high", 1.0)


def generate_queries(
    k: int,
    n: int,
    amp_low: float,
    amp_high: float,
    freq_low: float,
    freq_high: float,
    rng: np.random.Generator,
):
    amplitudes_matrix = rng.uniform(amp_low, amp_high, size=(k, n))
    frequencies_vector = rng.uniform(freq_low, freq_high, size=k)
    return amplitudes_matrix, frequencies_vector


def generate_real_data(n: int, lower: float, upper: float, rng: np.random.Generator):
    """
    Generate real data in the same way the standalone driver does:
    real_X ~ N(0, 1). The lower/upper bounds are unused here, since
    the base optimizer's generate_data() also uses a Gaussian for real_X
    when no real_data is passed.
    """
    return rng.normal(loc=0.0, scale=1.0, size=n)


def run_single(cfg: Dict, rng: np.random.Generator):
    n, k = cfg["n"], cfg["k"]

    # Booster-level theorem parameters
    eta = cfg["eta"]
    epsilon_sample = cfg["epsilon_sample"]    # sample-level ε for boosting theorem
    delta_sample = cfg["delta_sample"]        # sample-level δ for boosting theorem
    beta = cfg["beta"]

    tau, T_cap = cfg["tau"], cfg["max_iterations"]
    lower, upper = cfg["lower_bound"], cfg["upper_bound"]

    # Build synthetic query family and real data
    A, W = generate_queries(
        k,
        n,
        cfg["amp_low"],
        cfg["amp_high"],
        cfg["freq_low"],
        cfg["freq_high"],
        rng,
    )
    real_data = generate_real_data(n, lower, upper, rng)

    # Construct booster with λ, ρ, μ left as None → λ & ρ from base; μ from theorem with η as edge
    booster = SpikyBoostingAlgorithm(
        k=k,
        lambda_param=None,   # ← pulled from base optimizer
        eta=eta,             # ← used as edge γ in μ
        rho=None,            # ← pulled from base optimizer
        mu=None,             # ← computed from theorem
        T=T_cap,
        epsilon_sample=epsilon_sample,   # ← ε_sample for theorem (separate from base ε)
        delta=delta_sample,              # ← δ_sample for theorem (separate from base δ)
        beta=beta,
        n=n,
        upper_bound=upper,
        lower_bound=lower,
        tau=tau,
        mu_constant=float(cfg.get("mu_constant", 1.0)),
        min_rounds=int(cfg.get("min_rounds", 1)),
        stop_threshold_abs=cfg.get("stop_threshold_abs", None),
        theorem_T=int(cfg.get("theorem_T", 5)),  # NEW: pass theorem T
    )

    booster.set_queries(A, W)

    print("=== Testing Spiky Boosting Algorithm (λ,ρ from base; μ uses η as edge) ===")
    print(
        f"n={n} | |Q|={k} | "
        f"ε_sample={epsilon_sample} δ_sample={delta_sample} β={beta} η={eta} | T_cap={T_cap}"
    )
    if cfg.get("stop_threshold_abs") is not None:
        print(f"Absolute stop threshold: {cfg['stop_threshold_abs']:.6f}")
    if int(cfg.get("min_rounds", 1)) > 1:
        print(f"Minimum rounds before stop: {int(cfg['min_rounds'])}")
    print(
        "μ constant (C): "
        f"{float(cfg.get('mu_constant', 1.0)):.3f}"
    )
    print(f"Threshold scaling: thr = 0.500 * λ")

    results = booster.run_boosting(real_data, verbose=True)

    # Show how many iterations the base optimizer ran each boosting round
    print("\n--- Base optimizer iterations by round ---")
    for s in results["all_synopses"]:
        print(f"t={s['iteration']}: {s.get('base_iterations', 'unknown')}")

    # Evaluate final answers against truth (using the final ensemble median)
    final = results["final_answers"]
    errors = []
    for q in range(k):
        amps = A[q]
        w = W[q]
        real_ans = float(np.sum(real_data ** 2 + amps * np.sin(w * real_data)) / n)
        errors.append(abs(float(final[q]) - real_ans))
    errors = np.array(errors, dtype=float)

    # Threshold used for reporting ONLY (not for boosting logic)
    lam = float(booster.lambda_param)
    mu = float(booster.mu)
    thr = 0.5 * lam
    if cfg.get("stop_threshold_abs") is not None:
        thr = min(thr, float(cfg["stop_threshold_abs"]))

    print("\n=== Final Statistics ===")
    print(f"Early stopped: {results['early_stopped']} after {results['iterations_run']} iteration(s)")
    print(f"λ (from base)={lam:.6f}, μ (theorem w/ η edge)={mu:.6f}, threshold used={thr:.6f}")
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
        rng = np.random.default_rng(master_rng.integers(0, 2**63 - 1))
        _, errs, lam, mu, thr = run_single(CONFIG, rng)
        all_max.append(float(np.max(errs)))
        all_thr.append(float(thr))

    all_max = np.array(all_max, dtype=float)
    all_thr = np.array(all_thr, dtype=float)

    return True


if __name__ == "__main__":
    test_boosting()
