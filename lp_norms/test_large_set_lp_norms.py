"""
Main test script ONLY (no extra convergence/scaling runs).

It:
  - runs the large query set test
  - prints results
  - plots real_X datapoints after termination
  - plots initial guess vs fake_X with arrows showing movement
  - saves plots as PNGs

To change bounds later:
  - edit DEFAULT_* constants in lp_norms_oo.py
  - OR override LOWER/UPPER/PRECISION below in this file
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


from lp_norms_oo import run_lp_norms_optimization, DEFAULT_LOWER_BOUND, DEFAULT_UPPER_BOUND, DEFAULT_DATA_PRECISION

# Draw a fresh seed for this run and print it
# global_seed = np.random.default_rng().integers(0, 2**32 - 1)
global_seed = 1697329805
print(f"[Reproducibility] Seed for this run: {global_seed}")

# Optional overrides for this test (set to None to use defaults from lp_norms_oo.py)
LOWER_OVERRIDE = None       # e.g. -5.0
UPPER_OVERRIDE = None       # e.g.  5.0
PRECISION_OVERRIDE = None   # e.g. 2 for step=0.01, 3 for step=0.001


def test_large_query_set():
    print("=" * 80)
    print("TESTING LP NORMS COORDINATE DESCENT WITH LARGE QUERY SET")
    print("=" * 80)

    # Larger set of queries - p ranges from 1.1 to 5.0
    queries = np.round(np.linspace(1.2, 5.0, 20), 2).tolist()
    weights = [1.0 / len(queries)] * len(queries)

    print(f"Number of queries: {len(queries)}")
    print(f"Queries (p values): {queries}")
    print(f"Weights: {weights[:5]}... (all equal)")
    print(f"Sum of weights: {sum(weights):.6f}")

    # Parameters
    epsilon = 10.0
    delta = 1e-1
    beta = 0.05
    eta = 0.5
    n = 30
    tau = 0.01
    max_iterations = 2000
    theta = 0.6

    lower = DEFAULT_LOWER_BOUND if LOWER_OVERRIDE is None else LOWER_OVERRIDE
    upper = DEFAULT_UPPER_BOUND if UPPER_OVERRIDE is None else UPPER_OVERRIDE
    precision = DEFAULT_DATA_PRECISION if PRECISION_OVERRIDE is None else PRECISION_OVERRIDE

    print(f"\nParameters:")
    print(f"  epsilon: {epsilon}")
    print(f"  delta: {delta}")
    print(f"  beta: {beta}")
    print(f"  eta: {eta}")
    print(f"  n: {n}  (=> num_points = {2*n})")
    print(f"  tau: {tau}")
    print(f"  max_iterations: {max_iterations}")
    print(f"  bounds: [{lower}, {upper}]")
    print(f"  data_precision: {precision}  (grid step = {10**(-precision)})")
    print(f"  Target satisfaction ratio: {0.5 + eta}")
    print(f"  theta: {theta}")

    print("\n" + "=" * 80)
    print("RUNNING OPTIMIZATION")
    print("=" * 80)

    start_time = time.time()

    results = run_lp_norms_optimization(
        queries=queries,
        weights=weights,
        epsilon=epsilon,
        delta=delta,
        beta=beta,
        eta=eta,
        n=n,
        tau=tau,
        upper_bound=upper,
        lower_bound=lower,
        data_precision=precision,
        max_iterations=max_iterations,
        verbose=True,
        seed=global_seed,   # <- use the drawn seed for reproducibility
        theta=theta,
    )

    runtime = time.time() - start_time

    print("\n" + "=" * 80)
    print("OPTIMIZATION RESULTS")
    print("=" * 80)

    print(f"Runtime: {runtime:.2f} seconds")
    print(f"Total iterations: {results['num_iterations']}")
    print(f"Converged (loss update <= tau): {results['converged']}")
    print(f"Target reached (satisfaction >= 0.5 + eta): {results['target_reached']}")
    print(f"Both conditions met: {results['both_conditions_met']}")
    print(f"Initial weighted satisfaction: {results['initial_weighted_satisfaction']:.4f}")
    print(f"Final weighted satisfaction: {results['final_weighted_satisfaction']:.4f}")
    print(f"Target ratio: {results['target_ratio']:.4f}")
    print(f"Final loss: {results['final_loss']:.6f}")
    print(f"Final loss update: {results['final_loss_update']:.6f}")
    print(f"Lambda value: {results['lambda_val']:.6f}")
    threshold = (1 - theta) * results["lambda_val"]
    print(f"(1-theta)*Lambda: {threshold:.6f}")


    error_stats = results["error_stats"]
    print(f"\nError Statistics:")
    print(f"  Mean error: {error_stats['mean_error']:.6f}")
    print(f"  Max error: {error_stats['max_error']:.6f}")
    print(f"  Queries above (1-theta)*lambda: {error_stats['queries_above_one_minus_theta_lambda']}")
    print(f"  Percentage above (1-theta)*lambda: "
            f"{error_stats['queries_above_one_minus_theta_lambda'] / len(queries) * 100:.1f}%")


    print(f"\nIndividual Query Errors:")
    for i, (p, error) in enumerate(zip(queries, results["errors"])):
        status = "✓" if error < threshold else "✗"
        print(f"  Query {i + 1:2d} (p={p:4.2f}): {error:.6f} {status}")

    # ----------------------------------------------------------------------
    # Plot 1: Real database X only
    # ----------------------------------------------------------------------
    real_X = results["real_X"]
    idx = np.arange(len(real_X))

    # plt.figure(figsize=(11, 5))
    # plt.scatter(idx, real_X, s=14, label="real_X")
    # plt.title("Real Database X")
    # plt.xlabel("Coordinate index")
    # plt.ylabel("Value")
    # plt.legend()
    # plt.tight_layout()

    # out_png1 = "real_X_main_test.png"
    # plt.savefig(out_png1, dpi=200)
    # plt.show()
    # print(f"\nSaved plot to: {out_png1}")

    # ----------------------------------------------------------------------
    # Plot 2: Initial fake_X vs final fake_X, with arrows showing movement
    # ----------------------------------------------------------------------
    fake_X = results["fake_X"]
    fake_X_original = results["fake_X_original"]  # requires lp_norms_oo to store this
    real_X = results["real_X"]

    num_points = len(fake_X)
    x_axis = np.arange(1, num_points + 1)

    plt.figure(figsize=(14, 9))


    # Initial fake_X
    plt.scatter(
        x_axis,
        fake_X_original,
        s=np.full(num_points, 240),
        label=r"Initial guess $X_0$",
        color="deepskyblue",
        marker='o',
        facecolors='none',
        edgecolors='blue',
        linewidths=4.0,
    )
        # Final fake_X
    plt.scatter(
        x_axis,
        fake_X,
        s=np.full(num_points, 240),
        label=r"Private database $\hat{X}$",
        marker='o',
        color='deepskyblue',
    )

    # (Optional) real_X for reference
    plt.scatter(
        x_axis,
        real_X,
        s=np.full(num_points, 240),
        label=r"Real database $X$",
        color="orange",
    )

    # Arrows: color-coded by direction + large arrowheads
    # arrow_colors = np.where(fake_X > fake_X_original, "green", "red")
    arrow_colors = 'black'

    plt.quiver(
        x_axis,                        # x positions
        fake_X_original,               # y start
        np.zeros_like(x_axis),         # Δx = 0 (vertical arrows)
        fake_X - fake_X_original,      # Δy
        angles="xy",
        scale_units="xy",
        scale=1,
        color=arrow_colors,
        alpha=0.8,
        width=0.004,

        # Larger arrowheads (Option 1)
        headwidth=5,
        headlength=6,
        headaxislength=5,
    )

    from matplotlib.lines import Line2D

    arrow_legend_handles = [
        Line2D([0], [0], color="green", lw=2, marker=r"$\rightarrow$",
            markersize=12, label="Data value increased"),
        Line2D([0], [0], color="red", lw=2, marker=r"$\rightarrow$",
            markersize=12, label="Data value decreased"),
    ]


    # plt.legend(fontsize=20)

        # First legend: scatter points
    scatter_legend = plt.legend(fontsize=20, loc="upper right")

    # Add the scatter legend back explicitly
    plt.gca().add_artist(scatter_legend)

    # Second legend: arrows
    # plt.legend(
    #     handles=arrow_legend_handles,
    #     fontsize=20,
    #     loc="lower left",
    #     title_fontsize=20,
    # )

    plt.xlabel("Database Coordinates", fontsize=30)
    plt.ylabel("Data Value", fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    # plt.title("Initial Guess vs Final Private Database with Movement Arrows", fontsize=24)
    plt.tight_layout()

    out_png2 = "initial_vs_fake_with_arrows.png"
    plt.savefig(out_png2, dpi=200)
    plt.show()
    print(f"Saved plot to: {out_png2}")


       # ----------------------------------------------------------------------
    # Plot 3: Query outputs vs p-values (real, noisy, private)
    #         with Laplace noise visualized as vertical intervals
    # ----------------------------------------------------------------------
    p_vals = np.array(results["queries"])                     
    real_output = np.array(results["real_output"])            
    noisy_output = np.array(results["real_data_noisy_output"])
    private_output = np.array(results["fake_output"])         

    # Sort values by p
    sort_idx = np.argsort(p_vals)
    p_sorted = p_vals[sort_idx]
    real_sorted = real_output[sort_idx]
    noisy_sorted = noisy_output[sort_idx]
    private_sorted = private_output[sort_idx]



    plt.figure(figsize=(10, 8))

    # -------------------------------------------
    # Proxy artists for ONE clean combined legend
    # -------------------------------------------
    noise_bar_color = "deeppink"

    real_proxy = plt.Line2D([0], [0], marker="o", color="deepskyblue",
                            linestyle="-", markersize=10, label=r"Noiseless output on real data $q(X)$")

    private_proxy = plt.Line2D([0], [0], marker="^", color="orange",
                               linestyle="-", markersize=10, label=r"Noiseless output on synthetic data $q(\hat{X})$")

    noisy_proxy = plt.Line2D([0], [0], marker="o", color=noise_bar_color,
                             linestyle="None", markersize=10,
                             label=r"Noisy output on real data $\tilde{q}(X) = q(X)+\xi$")
    noise_bar_proxy = mpatches.Patch(facecolor=noise_bar_color,
                            alpha=0.25,
                            label=r"Added Laplace noise $\xi$")

    # -------------------------------------------
    # Plot real and synthetic outputs
    # -------------------------------------------
    plt.plot(
        p_sorted,
        real_sorted,
        "-o",
        linewidth=4,
        markersize=16,
        color="deepskyblue",
    )

    plt.plot(
        p_sorted,
        private_sorted,
        "-s",
        linewidth=4,
        markersize=16,
        color="orange",
    )

    # -------------------------------------------
    # Plot Laplace-noise intervals + noisy points
    # -------------------------------------------
    bar_half_width = 0.03

    for x, y_real, y_noisy in zip(p_sorted, real_sorted, noisy_sorted):
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

        # dashed connecting line
        plt.plot(
            [x, x],
            [y_real, y_noisy],
            linestyle="--",
            linewidth=1.5,
            color=noise_bar_color,
            alpha=0.7,
        )

        # noisy point itself
        plt.scatter(
            x, y_noisy,
            s=200,
            alpha=1,
            color=noise_bar_color,
        )

    # -------------------------------------------
    # Labels, ticks, legend
    # -------------------------------------------
    plt.xlabel(r"Values of $p$ of $\ell_p$-norm Queries", fontsize=24)
    plt.ylabel("Query Output", fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.grid(alpha=0.2)

    # Single combined legend
    plt.legend(
        handles=[real_proxy, private_proxy, noisy_proxy, noise_bar_proxy],
        fontsize=18,
        loc="best"
    )

    plt.tight_layout()

    out_png3 = "lp_query_outputs_vs_p.png"
    plt.savefig(out_png3, dpi=200)
    print(f"Saved plot to: {out_png3}")
    plt.show()



    return True


if __name__ == "__main__":
    print("Starting Large Query Set Test (ONLY main test; others suppressed)...\n")
    ok = test_large_query_set()

    print("\n" + "=" * 80)
    if ok:
        print("🎉 MAIN TEST PASSED! Plots generated after optimization.")
    else:
        print("❌ MAIN TEST FAILED!")
    print("=" * 80)
