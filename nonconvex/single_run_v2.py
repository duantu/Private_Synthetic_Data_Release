

#!/usr/bin/env python3
"""
single_run.py (v2)

Reproduce a *single* boosting run with a fixed RNG seed and save
everything needed for plotting later.

All saved files now have *_v2 appended to their names.
"""

import json
import pathlib
import sys
from typing import Any, Dict

import numpy as np

import test_boosting
from test_boosting import CONFIG as BOOST_CONFIG, run_single, Tee


# ---------------------------------------------------------------------
# CONFIG: which exact run to reproduce
# ---------------------------------------------------------------------
RUN_SEED: int = 5656766306055767438
# RUN_SEED: int = 8566110981845524785
RUN_INDEX: int = 0


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def to_serializable(obj: Any) -> Any:
    """Make objects JSON-serializable where possible."""
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def build_arrays_from_results(results: Dict[str, Any], errors: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Extract all the "nice for plotting" arrays from the run_single() outputs.
    """
    arrays: Dict[str, np.ndarray] = {}

    arrays["errors_final"] = np.asarray(errors, float)
    arrays["true_answers"] = np.asarray(results["true_answers"], float)
    arrays["final_distribution"] = np.asarray(results["final_distribution"], float)
    arrays["amplitudes"] = np.asarray(results["amplitudes"], float)
    arrays["frequencies"] = np.asarray(results["frequencies"], float)
    arrays["trig_flags"] = np.asarray(results["trig_flags"], bool)

    # NEW: pull out real_X from the first base_result (same across rounds)
    synopses = results["all_synopses"]
    if synopses:
        base0 = synopses[0]["base_result"]
        if "real_X" in base0:
            arrays["real_X"] = np.asarray(base0["real_X"], float)
        if "fake_X_original" in base0:
            arrays["X0"] = np.asarray(base0["fake_X_original"], float)  

    T = len(synopses)
    if T > 0:
        k = results["num_queries"]

        answers_by_round = np.zeros((T, k))
        errors_by_round = np.zeros((T, k))
        lambda_acc_by_round = np.zeros((T, k), bool)
        lambda_mu_acc_by_round = np.zeros((T, k), bool)
        base_iterations = np.zeros(T)
        accepted_attempt_idx = np.zeros(T)

        for t, s in enumerate(synopses):
            answers_by_round[t] = np.asarray(s["answers"], float)
            errors_by_round[t] = np.asarray(s["errors"], float)
            lambda_acc_by_round[t] = np.asarray(s["lambda_accurate"], bool)
            lambda_mu_acc_by_round[t] = np.asarray(s["lambda_mu_accurate"], bool)

            base_iterations[t] = (
                float(s["base_iterations"])
                if s.get("base_iterations") is not None
                else np.nan
            )
            accepted_attempt_idx[t] = (
                float(s.get("accepted_attempt"))
                if s.get("accepted_attempt") is not None
                else np.nan
            )

        arrays["answers_by_round"] = answers_by_round
        arrays["errors_by_round"] = errors_by_round
        arrays["lambda_accurate_by_round"] = lambda_acc_by_round
        arrays["lambda_mu_accurate_by_round"] = lambda_mu_acc_by_round
        arrays["base_iterations_by_round"] = base_iterations
        arrays["accepted_attempt_index_by_round"] = accepted_attempt_idx

    return arrays


def build_scalars_from_results(
    results: Dict[str, Any],
    lam: float,
    mu: float,
    thr: float,
) -> Dict[str, Any]:
    """Extract scalar-ish info for inspection and labels."""
    data: Dict[str, Any] = {
        "iterations_run": int(results["iterations_run"]),
        "early_stopped": bool(results["early_stopped"]),
        "lambda_param": float(lam),
        "mu": float(mu),
        "threshold_lambda_plus_mu": float(thr),
        "num_queries": int(results["num_queries"]),
        "upper_bound": float(results["upper_bound"]),
        "lower_bound": float(results["lower_bound"]),
        "offset": float(results["offset"]),
        "rho": float(results["rho"]) if results["rho"] is not None else None,
        "epsilon_base": float(results["epsilon_base"]),
        "delta_base": float(results["delta_base"]),
        "epsilon_sample": float(results["epsilon_sample"]),
        "delta_sample": float(results["delta_sample"]),
        "beta": float(results["beta"]),
        "eta": float(results["eta"]),
        "n": int(results["n"]),
    }
    return data


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main() -> None:
    # --------------------------------------------------------------
    # 1. Create output directory
    # --------------------------------------------------------------
    out_dir = pathlib.Path("single_run_output_v2") / f"seed_{RUN_SEED}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------------
    # 2. Replace stdout/stderr with a Tee to single_run_v2.log
    # --------------------------------------------------------------
    log_path = out_dir / "single_run_v2.log"
    log_f = open(log_path, "w")

    sys.stdout = Tee(sys.__stdout__, log_f)
    sys.stderr = Tee(sys.__stderr__, log_f)

    print("========== Single Run Boosting Reproduction (v2) ==========")
    print(f"Run seed: {RUN_SEED}")
    print(f"Logging to: {log_path}")
    print("===========================================================")

    # --------------------------------------------------------------
    # 3. Clone CONFIG and seed RNG
    # --------------------------------------------------------------
    config: Dict[str, Any] = dict(BOOST_CONFIG)
    config["num_runs"] = 1  # run_single doesn't use it

    rng = np.random.default_rng(RUN_SEED)

    # --------------------------------------------------------------
    # 4. Run one boosting experiment
    # --------------------------------------------------------------
    results, errors, lam, mu, thr = run_single(config, rng, run_seed=RUN_SEED)

    # --------------------------------------------------------------
    # 5. Save metadata (renamed)
    # --------------------------------------------------------------
    metadata = {
        "run_seed": RUN_SEED,
        "run_index": RUN_INDEX,
        "config_used": to_serializable(config),
    }
    with open(out_dir / "metadata_v2.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # --------------------------------------------------------------
    # 6. Save arrays and scalars (renamed)
    # --------------------------------------------------------------
    arrays_dict = build_arrays_from_results(results, np.asarray(errors, float))
    scalars_dict = build_scalars_from_results(results, lam, mu, thr)

    # NEW: save real_X separately for super easy loading
    if "real_X" in arrays_dict:
        np.save(out_dir / "real_X_v2.npy", arrays_dict["real_X"])

    if "X0" in arrays_dict:
        np.save(out_dir / "X0_same_as_run.npy", arrays_dict["X0"])   


    np.savez_compressed(out_dir / "single_run_results_arrays_v2.npz", **arrays_dict)

    with open(out_dir / "single_run_results_scalars_v2.json", "w") as f:
        json.dump(to_serializable(scalars_dict), f, indent=2)

    with open(out_dir / "single_run_results_raw_v2.json", "w") as f:
        json.dump(to_serializable(results), f, indent=2)

    print("Saved (v2):")
    print(" - single_run_v2.log")
    print(" - metadata_v2.json")
    print(" - single_run_results_arrays_v2.npz")
    print(" - single_run_results_scalars_v2.json")
    print(" - single_run_results_raw_v2.json")
    print(" - real_X_v2.npy")  # NEW
    print("===========================================================")


if __name__ == "__main__":
    main()
