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

import sys

class Tee:
    """
    Simple tee-style stream: writes to multiple underlying streams.
    """
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()  # ensure it appears promptly

    def flush(self):
        for s in self.streams:
            s.flush()

# Open/overwrite run.log on every run
log_file = open("run.log", "w")

# Send all prints to BOTH the terminal and run.log
sys.stdout = Tee(sys.__stdout__, log_file)
sys.stderr = Tee(sys.__stderr__, log_file)



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
CONFIG.setdefault("theorem_T", 10)              # T used in μ theorem. This is an arbitrary number decided based on observation

# --- Booster's own ε_sample and δ_sample (PURELY boosting-level) ---
# These DO NOT default to the base ε, δ. Think of them as separate.
CONFIG.setdefault("epsilon_sample", 1)       # sample-level ε for the theorem
CONFIG.setdefault("delta_sample", 1e-1)        # sample-level δ for the theorem
CONFIG.setdefault("mu", 0)                   # desired μ (excess over λ)
CONFIG.setdefault("max_base_attempts", 30)

CONFIG["theta"] = 0.1



# Just in case you ever remove any of these from BASE_CONFIG in the future,
# keep safe defaults here.
CONFIG["max_iterations"] = int(math.ceil(math.log(CONFIG["k"]) / (CONFIG["eta"] ** 2))) # max number of boosting rounds needed, by Theorem 6.1
CONFIG["tau"] = 1e-10
CONFIG["upper_bound"] = math.pi
CONFIG["lower_bound"] = -math.pi

# You can now freely change these here; they WILL override BASE_CONFIG.
CONFIG["freq_low"] = 1.2
CONFIG["freq_high"] = 2.0
CONFIG["amp_low"] = 0.1
CONFIG["amp_high"] = 1.0
CONFIG["offset"] = 0.0

# Mixture of "easy" and "hard" queries (both amplitude and frequency)
CONFIG["amp_mid"] = 0.5  # split point between easy and hard amplitudes


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
        * amplitudes in [0.4, 0.6]
        * frequencies in [2.5, 3.0]          (moderate frequency, always nonconvex but mild)

    - HARD queries (fraction p_hard):
        * amplitudes in [0.5, 1.0]
        * frequencies in [6.0, 8.0]          (much higher frequency → harder nonconvexity)

    trig_flags:
        - alternate sin/cos within each group, then shuffle.
    """

    # Mixture controls from CONFIG
    p_hard = 0.05    # fraction of hard queries
    

    k_hard = int(round(p_hard * k))
    k_easy = k - k_hard

    amplitudes_matrix = np.zeros((k, n), dtype=float)

    # ---------------- EASY QUERIES ----------------
    if k_easy > 0:
        # a_{j,i} ~ Unif[0.4, 0.6]
        amplitudes_matrix[:k_easy, :] = rng.uniform(
            0.36, 0.55, size=(k_easy, n)
        )
        # w_j ~ Unif[2.5, 3.0]
        freqs_easy = rng.uniform(2.4, 2.9, size=k_easy)
    else:
        freqs_easy = np.array([], dtype=float)

    # ---------------- HARD QUERIES ----------------
    if k_hard > 0:
        # a_{j,i} ~ Unif[0.6, 1.0]
        amplitudes_matrix[k_easy:, :] = rng.uniform(
            0.45, 0.9, size=(k_hard, n)
        )
        # w_j ~ Unif[4, 6]
        freqs_hard = rng.uniform(5.5, 7, size=k_hard)
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
    Generate real data as N(0,1) *conditioned* on [lower, upper],
    by resampling any coordinates that fall outside the interval.
    """
    real_X = np.empty(n, dtype=float)
    filled = 0
    while filled < n:
        batch = rng.normal(loc=0.0, scale=1.0, size=n - filled)
        mask = (batch >= lower) & (batch <= upper)
        if not np.any(mask):
            continue
        good = batch[mask]
        n_good = good.shape[0]
        real_X[filled : filled + n_good] = good
        filled += n_good
    return real_X



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

    theta = cfg["theta"]


    # Sensitivity ρ: match SpikyNonconvexCoordinateDescent's formula
    rho = (upper ** 2 + 1.0) / n

    # Choose the theorem horizon T (you can tune theorem_T in CONFIG)
    T_theorem = int(cfg.get("theorem_T", 100))

    # ε_sample from the new theorem:
    # epsilon_sample = compute_epsilon_sample_from_theorem(
    #     rho=rho,
    #     k=k,
    #     delta_sample=delta_sample,
    #     gamma_margin=eta,
    #     T=T_theorem,
    #     mu=mu,
    # )
    epsilon_sample = 0

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
        max_base_attempts=int(cfg.get("max_base_attempts", 50)),
        repro_seed=run_seed,
        theta=theta, 
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
    """
    Run boosting multiple times.
    Each run depends *only* on its single run_seed.
    No global seed needed.
    """
    base_seed = CONFIG.get("seed", None)

    if base_seed is None:
        # if no seed provided, randomly choose a base for the run sequence
        base_seed = np.random.randint(0, 2**63 - 1)
        print(f"No seed provided — using base_seed = {base_seed}")
    else:
        print(f"Using provided base seed = {base_seed}")

    all_max: List[float] = []
    all_thr: List[float] = []

    # NEW: keep a small per-run summary in memory
    run_summaries: List[Dict] = []

    for run in range(CONFIG["num_runs"]):

        # Each run uses exactly this seed — reproducible by itself
        run_seed = base_seed + run

        print(f"\n========== RUN {run + 1} / {CONFIG['num_runs']} ==========")
        print(f"[REPRO] run_seed = {run_seed}")

        rng = np.random.default_rng(run_seed)

        try:
            results, errs, lam, mu, thr = run_single(CONFIG, rng, run_seed=run_seed)
        except BaseSynopsisFailed as e:
            print(f"[RUN FAILED] {e}")
            # record a failed run in the summary too
            run_summaries.append(
                {
                    "run_index": run + 1,
                    "run_seed": run_seed,
                    "status": "failed",
                    "iterations_run": 0,
                    "early_stopped": False,
                    "lambda": None,
                    "mu": None,
                    "threshold": None,
                    "max_error": None,
                    "share_within_threshold": None,
                }
            )
            continue

        all_max.append(float(np.max(errs)))
        all_thr.append(float(thr))

        iterations_run = int(results["iterations_run"])
        early_stopped = bool(results["early_stopped"])
        max_err = float(np.max(errs))
        share_within = float(np.mean(errs <= thr))

        # record successful run summary
        run_summaries.append(
            {
                "run_index": run + 1,
                "run_seed": run_seed,
                "status": "ok",
                "iterations_run": iterations_run,
                "early_stopped": early_stopped,
                "lambda": float(lam),
                "mu": float(mu),
                "threshold": float(thr),
                "max_error": max_err,
                "share_within_threshold": share_within,
            }
        )

    all_max = np.array(all_max, dtype=float) if all_max else np.array([])
    all_thr = np.array(all_thr, dtype=float) if all_thr else np.array([])

    # ------------- PRINT A QUICK HUMAN-READABLE SUMMARY -------------
    print("\n================ BOOSTING RUN SUMMARY ================")
    print("run  | iterations | early_stop | max_err   | share<=thr | status")
    print("-----+-----------+------------+----------+------------+--------")
    for s in run_summaries:
        if s["status"] != "ok":
            print(
                f"{s['run_index']:>3d}  |"
                f" {s['iterations_run']:>9d} |"
                f" {'-':>10} |"
                f" {'-':>8} |"
                f" {'-':>10} |"
                f" {s['status']}"
            )
        else:
            print(
                f"{s['run_index']:>3d}  |"
                f" {s['iterations_run']:>9d} |"
                f" {str(s['early_stopped']):>10} |"
                f" {s['max_error']:.6f} |"
                f" {s['share_within_threshold']:.3f} |"
                f" {s['status']}"
            )

    # Highlight the interesting ones: runs with > 1 boosting iteration
    interesting = [s for s in run_summaries if s["status"] == "ok" and s["iterations_run"] > 1]
    print("\nRuns with MORE THAN 1 boosting iteration:")
    if not interesting:
        print("  (none)")
    else:
        for s in interesting:
            print(
                f"  run {s['run_index']} (seed={s['run_seed']}): "
                f"{s['iterations_run']} iterations, "
                f"max_err={s['max_error']:.6f}, "
                f"share<=thr={s['share_within_threshold']:.3f}"
            )

    # # ------------- OPTIONAL: WRITE A CSV SUMMARY TO DISK -------------
    # outdir = "./boosting_logs"
    # os.makedirs(outdir, exist_ok=True)
    # summary_path = os.path.join(outdir, "boosting_run_summary.csv")

    # with open(summary_path, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(
    #         [
    #             "run_index",
    #             "run_seed",
    #             "status",
    #             "iterations_run",
    #             "early_stopped",
    #             "lambda",
    #             "mu",
    #             "threshold",
    #             "max_error",
    #             "share_within_threshold",
    #         ]
    #     )
    #     for s in run_summaries:
    #         writer.writerow(
    #             [
    #                 s["run_index"],
    #                 s["run_seed"],
    #                 s["status"],
    #                 s["iterations_run"],
    #                 s["early_stopped"],
    #                 s["lambda"],
    #                 s["mu"],
    #                 s["threshold"],
    #                 s["max_error"],
    #                 s["share_within_threshold"],
    #             ]
    #         )

    # print(f"\nSummary written to: {summary_path}")

    return True



if __name__ == "__main__":
    test_boosting()
