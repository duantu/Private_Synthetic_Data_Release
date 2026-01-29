#!/usr/bin/env python3
"""
Plot initial guess X_0, real database X, and final synthetic database \hat{X}
for a single spiky-boosting run.
"""

import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Base directory for THIS run
# ---------------------------------------------------------------------
BASE_DIR = pathlib.Path(
    "/home/duantu/Documents/UIC/Research/Private_Synthetic_Data_Release/"
    "code/nonconvex/single_run_output_v2/seed_5656766306055767438"
)

RAW_RESULTS_PATH = BASE_DIR / "single_run_results_raw_v2.json"
X0_PATH = BASE_DIR / "X0_same_as_run.npy"  # <- must exist for X_0 to be plotted
OUT_FIG = BASE_DIR / "X0_realX_fakeX_spikyboosting.png"


def main() -> None:
    # --------------------------------------------------------------
    # 1. Load raw JSON (contains all synopses, including real_X & fake_X)
    # --------------------------------------------------------------
    if not RAW_RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"Could not find results JSON at:\n  {RAW_RESULTS_PATH}\n"
            f"Check that single_run.py (v2) finished and saved this file."
        )

    with open(RAW_RESULTS_PATH, "r") as f:
        raw = json.load(f)

    # Take real_X and fake_X from the *final* synopsis
    try:
        syn_last = raw["all_synopses"][-1]["base_result"]
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(
            "Could not access raw['all_synopses'][-1]['base_result'].\n"
            "Double-check the structure of single_run_results_raw_v2.json."
        ) from e

    real_X = np.asarray(syn_last["real_X"], dtype=float)
    fake_X = np.asarray(syn_last["fake_X"], dtype=float)

    # --------------------------------------------------------------
    # 2. Load X_0 (initial fake data) if available
    # --------------------------------------------------------------
    if not X0_PATH.is_file():
        raise FileNotFoundError(
            f"X_0 file not found at:\n  {X0_PATH}\n\n"
            "You need to save self.fake_X_original (or X0) in your pipeline,\n"
            "e.g., as 'X0_same_as_run.npy' in single_run_output_v2.\n"
        )

    X0 = np.load(X0_PATH)

    # --------------------------------------------------------------
    # 3. Shape checks
    # --------------------------------------------------------------
    if X0.shape != fake_X.shape or real_X.shape != fake_X.shape:
        raise ValueError(
            "Shape mismatch between X_0, real_X, and fake_X:\n"
            f"  X_0 shape    = {X0.shape}\n"
            f"  real_X shape = {real_X.shape}\n"
            f"  fake_X shape = {fake_X.shape}\n"
            "All three must be the same length."
        )

    n = len(real_X)
    x_axis = np.arange(1, n + 1)

    print(f"Loaded vectors of length n = {n}")
    print("  X_0    :", X0.shape)
    print("  real_X :", real_X.shape)
    print("  fake_X :", fake_X.shape)

    # --------------------------------------------------------------
    # 4. Plot X_0, real_X, fake_X
    # --------------------------------------------------------------
    plt.figure(figsize=(14, 9))

    # Initial guess X_0 (hollow blue markers)
    plt.scatter(
        x_axis,
        X0,
        s=np.full(n, 40),
        label=r"Initial guess $X_0$",
        facecolors="none",
        edgecolors="blue",
        linewidths=2,
    )

    # Final synthetic database \hat X (solid blue)
    plt.scatter(
        x_axis,
        fake_X,
        s=np.full(n, 40),
        label=r"Private database $\hat{X}$",
        color="deepskyblue",
        alpha=1,
    )

    # Real database X (orange)
    plt.scatter(
        x_axis,
        real_X,
        s=np.full(n, 40),
        label=r"Real database $X$",
        color="tab:orange",
        alpha=1,
    )

    # If you want arrows from X_0 to \hat X, uncomment this block:
    # plt.quiver(
    #     x_axis,                    # x positions
    #     X0,                        # y start (X_0)
    #     np.zeros_like(x_axis),     # Δx = 0 (vertical arrows)
    #     fake_X - X0,               # Δy
    #     angles="xy",
    #     scale_units="xy",
    #     scale=1,
    #     color="black",
    #     alpha=0.4,
    #     width=0.0018,
    # )

    plt.legend(
        fontsize=18,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
    )

    plt.xlabel("Database coordinates", fontsize=26)
    plt.ylabel("Data value", fontsize=26)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.tight_layout()

    plt.savefig(OUT_FIG, dpi=200)
    plt.show()

    print(f"Saved figure to: {OUT_FIG}")


if __name__ == "__main__":
    main()
