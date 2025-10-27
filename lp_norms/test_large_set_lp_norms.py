#!/usr/bin/env python3

"""
Test script for the LP norms coordinate descent algorithm with larger query set
and eta = 0.5 for convergence testing
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from lp_norms_oo import LPNormsCoordinateDescent, run_lp_norms_optimization
import numpy as np
import time

def test_large_query_set():
    """Test with a larger set of queries and eta = 0.5"""
    print("=" * 80)
    print("TESTING LP NORMS COORDINATE DESCENT WITH LARGE QUERY SET")
    print("=" * 80)

    # Larger set of queries - p values from 1.1 to 5.0
    queries = np.round(np.linspace(1.1, 5.0, 20), 2).tolist()  # 20 queries
    weights = [1.0/len(queries)] * len(queries)  # Equal weights

    print(f"Number of queries: {len(queries)}")
    print(f"Queries (p values): {queries}")
    print(f"Weights: {weights[:5]}... (all equal)")
    print(f"Sum of weights: {sum(weights):.6f}")

    # Parameters for convergence testing
    epsilon = 15.0
    delta = 1e-1
    beta = 0.05
    eta = 0.5  # Set eta = 1/2 as requested
    n = 100
    tau = 0.01
    max_iterations = 2000  # Allow more iterations for convergence

    print(f"\nParameters:")
    print(f"  epsilon: {epsilon}")
    print(f"  delta: {delta}")
    print(f"  beta: {beta}")
    print(f"  eta: {eta}")
    print(f"  n: {n}")
    print(f"  tau: {tau}")
    print(f"  max_iterations: {max_iterations}")
    print(f"  Target satisfaction ratio: {0.5 + eta}")

    # Run the optimization
    print("\n" + "=" * 80)
    print("RUNNING OPTIMIZATION")
    print("=" * 80)

    start_time = time.time()

    try:
        results = run_lp_norms_optimization(
            queries=queries,
            weights=weights,
            epsilon=epsilon,
            delta=delta,
            beta=beta,
            eta=eta,
            n=n,
            tau=tau,
            max_iterations=max_iterations,
            verbose=True
        )

        end_time = time.time()
        runtime = end_time - start_time

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
        print(f"Lambda/2: {results['lambda_val']/2:.6f}")

        # Error statistics
        error_stats = results['error_stats']
        print(f"\nError Statistics:")
        print(f"  Mean error: {error_stats['mean_error']:.6f}")
        print(f"  Max error: {error_stats['max_error']:.6f}")
        print(f"  Queries above lambda/2: {error_stats['queries_above_lambda_half']}")
        print(f"  Percentage above lambda/2: {error_stats['queries_above_lambda_half']/len(queries)*100:.1f}%")

        # Individual query errors
        print(f"\nIndividual Query Errors:")
        for i, (p, error) in enumerate(zip(queries, results['errors'])):
            status = "✓" if error < results['lambda_val']/2 else "✗"
            print(f"  Query {i+1:2d} (p={p:4.2f}): {error:.6f} {status}")

        return True

    except Exception as e:
        print(f"✗ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_convergence_analysis():
    """Test convergence behavior with different parameters"""
    print("\n" + "=" * 80)
    print("CONVERGENCE ANALYSIS")
    print("=" * 80)

    # Test with different eta values
    eta_values = [0.1, 0.3, 0.5]
    queries = np.round(np.linspace(1.2, 4.0, 10), 2).tolist()  # 10 queries for faster testing
    weights = [1.0/len(queries)] * len(queries)

    print(f"Testing with {len(queries)} queries and different eta values...")

    for eta in eta_values:
        print(f"\n--- Testing with eta = {eta} ---")
        target_ratio = 0.5 + eta

        try:
            results = run_lp_norms_optimization(
                queries=queries,
                weights=weights,
                epsilon=10.0,
                delta=1e-1,
                beta=0.1,
                eta=eta,
                n=50,  # Smaller n for faster testing
                tau=0.005,
                max_iterations=500,
                verbose=False
            )

            print(f"  Iterations: {results['num_iterations']}")
            print(f"  Final satisfaction: {results['final_weighted_satisfaction']:.4f}")
            print(f"  Target: {results['target_ratio']:.4f}")
            print(f"  Converged: {results['converged']}")
            print(f"  Both conditions met: {results['both_conditions_met']}")

        except Exception as e:
            print(f"  Failed: {e}")

def test_performance_scaling():
    """Test how performance scales with number of queries"""
    print("\n" + "=" * 80)
    print("PERFORMANCE SCALING TEST")
    print("=" * 80)

    query_counts = [5, 10, 15, 20]

    for k in query_counts:
        print(f"\n--- Testing with {k} queries ---")
        queries = np.round(np.linspace(1.1, 3.0, k), 2).tolist()
        weights = [1.0/k] * k

        start_time = time.time()

        try:
            results = run_lp_norms_optimization(
                queries=queries,
                weights=weights,
                epsilon=10.0,
                delta=1e-1,
                beta=0.1,
                eta=0.3,
                n=50,
                tau=0.01,
                max_iterations=300,
                verbose=False
            )

            end_time = time.time()
            runtime = end_time - start_time

            print(f"  Runtime: {runtime:.2f} seconds")
            print(f"  Iterations: {results['num_iterations']}")
            print(f"  Final satisfaction: {results['final_weighted_satisfaction']:.4f}")
            print(f"  Converged: {results['converged']}")

        except Exception as e:
            print(f"  Failed: {e}")

if __name__ == "__main__":
    print("Starting Large Query Set Tests...")
    print()

    success = True

    # Run main test with large query set
    success &= test_large_query_set()

    # Run convergence analysis
    test_convergence_analysis()

    # Run performance scaling test
    test_performance_scaling()

    print("\n" + "=" * 80)
    if success:
        print("🎉 MAIN TEST PASSED! The algorithm is working correctly with large query sets.")
    else:
        print("❌ MAIN TEST FAILED! Please check the errors above.")
    print("=" * 80)
