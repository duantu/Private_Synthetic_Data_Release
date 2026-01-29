#!/usr/bin/env python3
r"""
Plot perturbed-parabola query outputs:

- x-axis: query indices (1..k), ordered as:
    [ all sin queries by increasing frequency, then all cos queries by increasing frequency ]
- y-axis: query output values

Curves / visuals:
- Blue circles: noiseless output on real data, q(X)
- Three colored triangle lines: for each boosting iteration t,
    the median-of-synopses synthetic outputs "answers" at that iteration
- Pink bar: Laplace noise interval [q(X), q(X) + ξ]
"""

import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

# ---------------------------------------------------------------------
# Base directory for THIS run
# ---------------------------------------------------------------------
BASE_DIR = pathlib.Path(
    "/home/duantu/Documents/UIC/Research/Private_Synthetic_Data_Release/"
    "code/nonconvex/single_run_output_v2/seed_5656766306055767438"
)

RAW_RESULTS_PATH = BASE_DIR / "single_run_results_raw_v2.json"
OUT_FIG = BASE_DIR / "parabola_queries_outputs_noise_answers_per_round.png"


def main() -> None:
    # --------------------------------------------------------------
    # 1. Load raw JSON from the single run
    # --------------------------------------------------------------
    if not RAW_RESULTS_PATH.is_file():
        raise FileNotFoundError(f"Results JSON not found at:\n  {RAW_RESULTS_PATH}")

    with open(RAW_RESULTS_PATH, "r") as f:
        raw = json.load(f)

    synopses = raw["all_synopses"]
    T = len(synopses)
    if T < 1:
        raise RuntimeError("No synopses found in results['all_synopses'].")

    # we know there are 3 boosting iterations, but be safe:
    max_rounds_to_plot = 3
    R = min(max_rounds_to_plot, T)

    # base_result of round 1 (t=1, index 0) – use for real_output and Laplace noise
    base0 = synopses[0]["base_result"]

    real_output = np.asarray(base0["real_output"], dtype=float)              # q(X)
    lap_noise   = np.asarray(base0["lap_noise"], dtype=float)                # ξ
    noisy_output = real_output + lap_noise                                   # q(X) + ξ

    # frequencies and trig_flags are stored at top-level in your results
    frequencies = np.asarray(raw["frequencies"], dtype=float)               # shape (k,)
    trig_flags  = np.asarray(raw["trig_flags"], dtype=bool)                 # False=sin, True=cos

    k = real_output.shape[0]
    if frequencies.shape[0] != k or trig_flags.shape[0] != k:
        raise ValueError(
            "Mismatch among k for real_output, frequencies, trig_flags:\n"
            f"  len(real_output) = {k}\n"
            f"  len(frequencies) = {frequencies.shape[0]}\n"
            f"  len(trig_flags)  = {trig_flags.shape[0]}"
        )

    # --------------------------------------------------------------
    # 2. Collect per-iteration "answers" (median-of-synopses outputs)
    # --------------------------------------------------------------
    # answers_rounds[t, j] = synthetic answer for query j after boosting iteration t+1
    answers_rounds = []
    for t in range(R):
        s_t = synopses[t]
        ans_t = np.asarray(s_t["answers"], dtype=float)
        if ans_t.shape[0] != k:
            raise ValueError(
                f"Round {t+1} 'answers' has length {ans_t.shape[0]}, expected {k}."
            )
        answers_rounds.append(ans_t)

    answers_rounds = np.stack(answers_rounds, axis=0)  # shape: (R, k)

    # --------------------------------------------------------------
    # 3. Build ordering: sin-queries first (by freq), then cos-queries (by freq)
    # --------------------------------------------------------------
    indices = np.arange(k)

    sin_idx = indices[~trig_flags]   # trig_flags == False → sin
    cos_idx = indices[trig_flags]    # trig_flags == True  → cos

    sin_sorted = sin_idx[np.argsort(frequencies[sin_idx])]
    cos_sorted = cos_idx[np.argsort(frequencies[cos_idx])]
    order = np.concatenate([sin_sorted, cos_sorted])

    # reorder all arrays
    x_positions   = np.arange(1, k + 1)                 # 1..k
    x_ordered     = x_positions
    real_sorted   = real_output[order]
    noisy_sorted  = noisy_output[order]

    answers_sorted = answers_rounds[:, order]           # shape (R, k)

    # The boundary between sin and cos groups (for optional visual separator)
    n_sin = sin_sorted.shape[0]

    # --------------------------------------------------------------
    # 4. Make the plot
    # --------------------------------------------------------------
    plt.figure(figsize=(12, 8))

    noise_bar_color = "deeppink"

    # --- Plot q(X) (real, noiseless) ---
    real_line = plt.plot(
        x_ordered,
        real_sorted,
        "-o",
        linewidth=3,
        markersize=10,
        color="deepskyblue",
        label=r"Noiseless output on real data $q(X)$",
    )[0]

    # --- Plot per-iteration synthetic answers ---
    cmap = cm.get_cmap("Oranges")
    answer_lines = []
    for t in range(R):
        color_t = cmap(0.4 + 0.5 * t / max(1, R - 1))  # from lighter to darker orange
        line_t = plt.plot(
            x_ordered,
            answers_sorted[t],
            "-^",
            linewidth=2,
            markersize=8,
            color=color_t,
            label=rf"Synthetic answers after round {t+1}",
        )[0]
        answer_lines.append(line_t)

    # --- Plot Laplace noise intervals as pink vertical bars ---
    bar_half_width = 0.25  # x-width of each noise bar (tune if needed)

    for x, y_real, y_noisy in zip(x_ordered, real_sorted, noisy_sorted):
        y_min, y_max = sorted((y_real, y_noisy))

        # translucent noise bar
        plt.fill_between(
            [x - bar_half_width, x + bar_half_width],
            [y_min, y_min],
            [y_max, y_max],
            color=noise_bar_color,
            alpha=0.25,
            edgecolor="none",
        )

        # optional dashed line to indicate direction
        plt.plot(
            [x, x],
            [y_real, y_noisy],
            linestyle="--",
            linewidth=1.2,
            color=noise_bar_color,
            alpha=0.7,
        )

    # Separate handle for the noise bar in the legend
    noise_bar_proxy = mpatches.Patch(
        facecolor=noise_bar_color,
        alpha=0.25,
        label=r"Laplace noise interval $[q(X),\, q(X)+\xi]$",
    )

    # --- Optional: vertical line separating sin and cos groups ---
    if 0 < n_sin < k:
        plt.axvline(x=n_sin + 0.5, color="gray", linestyle=":", linewidth=1.5)
        # You can comment these out if you don't want labels at the top
        plt.text(
            n_sin / 2 + 0.5,
            plt.ylim()[1],
            "sin",
            ha="center",
            va="bottom",
            fontsize=12,
        )
        plt.text(
            n_sin + (k - n_sin) / 2 + 0.5,
            plt.ylim()[1],
            "cos",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    # --- Labels, ticks, legend ---
    plt.xlabel(
        "Query index (sin queries first, then cos; each group sorted by frequency)",
        fontsize=18,
    )
    plt.ylabel("Query output value", fontsize=20)
    plt.xticks(x_ordered, x_ordered, fontsize=12)
    plt.yticks(fontsize=14)
    plt.grid(alpha=0.2)

    # Build legend: real line, all per-round synthetic lines, and noise bar proxy
    handles = [real_line] + answer_lines + [noise_bar_proxy]
    plt.legend(
        handles=handles,
        fontsize=12,
        loc="best",
    )

    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=200)
    plt.show()

    print(f"Saved figure to: {OUT_FIG}")
    print(f"Number of queries: {k} (sin: {n_sin}, cos: {k - n_sin})")
    print(f"Boosting rounds plotted (answers): {R}")


if __name__ == "__main__":
    main()
