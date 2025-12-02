#!/usr/bin/env python3
"""
Simple example script showing how to use the separated testing framework.
"""

from test_spiky_nonconvex import test_spiky_nonconvex_queries, test_multiple_runs, test_parameter_sweep

def main():
    print("=== Spiky Nonconvex Testing Examples ===\n")
    
    # Example 1: Single test with custom parameters
    print("1. Single test with custom parameters:")
    result = test_spiky_nonconvex_queries(
        n=100,           # Number of data points
        epsilon=2.0,     # Privacy parameter
        k=15,            # Number of queries
        max_iterations=50,
        verbose=True
    )
    print(f"Result: Error={result['final_error']:.4f}, Satisfaction={result['final_satisfaction']:.4f}\n")
    
    # Example 2: Multiple runs with different random seeds
    print("2. Multiple runs with different initializations:")
    results = test_multiple_runs(
        num_runs=5,
        n=150,
        epsilon=1.5,
        k=12,
        max_iterations=30,
        verbose=False
    )
    print()
    
    # Example 3: Quick parameter sweep
    print("3. Quick parameter sweep:")
    print("Testing different epsilon values...")
    for eps in [0.5, 1.0, 2.0, 4.0]:
        result = test_spiky_nonconvex_queries(
            n=200, epsilon=eps, k=10, verbose=False
        )
        print(f"ε={eps}: Error={result['final_error']:.4f}, Satisfaction={result['final_satisfaction']:.4f}")
    
    print("\n=== Testing Complete ===")

if __name__ == "__main__":
    main()
