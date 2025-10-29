#!/usr/bin/env python3
"""
Test script for the spiky boosting algorithm.
"""

import numpy as np
import math
from spiky_boosting import SpikyBoostingAlgorithm

def test_boosting():
    """Test the boosting algorithm with spiky nonconvex queries."""
    print("=== Testing Spiky Boosting Algorithm ===\n")
    
    # Parameters
    n = 100  # Number of data points
    num_total_queries = 50  # Total queries in set Q
    k = 10  # Number of queries to sample per iteration
    T = 5  # Number of boosting iterations
    
    # Generate query set Q
    np.random.seed(42)
    amplitudes_matrix = np.random.uniform(0.1, 1.0, (num_total_queries, n))
    frequencies_vector = np.random.uniform(1.0, 10.0, num_total_queries)
    
    # Generate real data
    real_data = np.random.choice(
        np.arange(-math.pi, math.pi + 0.01, 0.01), 
        n, replace=False
    )
    
    print(f"Parameters:")
    print(f"  n = {n}")
    print(f"  Total queries in Q = {num_total_queries}")
    print(f"  Queries sampled per iteration k = {k}")
    print(f"  Boosting iterations T = {T}")
    print()
    
    # Create boosting algorithm
    booster = SpikyBoostingAlgorithm(
        k=k,
        lambda_param=0.5,  # λ parameter
        eta=0.01,  # Edge parameter
        rho=1.0,  # Sensitivity (not directly used in current implementation)
        mu=0.2,  # Margin for accuracy scoring
        T=T,
        epsilon=1.0,  # DP parameter
        delta=1e-5,  # DP parameter
        beta=0.1,  # Failure probability
        n=n,
        upper_bound=math.pi,
        lower_bound=-math.pi
    )
    
    # Set queries
    booster.set_queries(amplitudes_matrix, frequencies_vector)
    
    # Run boosting
    results = booster.run_boosting(real_data, verbose=True)
    
    # Analyze results
    print("\n=== Results Analysis ===")
    final_answers = results['final_answers']
    
    # Compute errors for final boosted answers
    errors = []
    for query_idx in range(num_total_queries):
        # Real answer
        amps = amplitudes_matrix[query_idx]
        freq = frequencies_vector[query_idx]
        real_sum = np.sum(real_data**2 + amps * np.sin(freq * real_data))
        real_answer = real_sum / n
        
        # Boosted answer
        boosted_answer = final_answers[query_idx]
        
        # Error
        error = abs(boosted_answer - real_answer)
        errors.append(error)
    
    print(f"Final boosted answer statistics:")
    print(f"  Mean error: {np.mean(errors):.6f}")
    print(f"  Median error: {np.median(errors):.6f}")
    print(f"  Max error: {np.max(errors):.6f}")
    print(f"  Min error: {np.min(errors):.6f}")
    print(f"  Queries with error < 0.5: {np.sum(np.array(errors) < 0.5)} / {num_total_queries}")
    
    # Compare with single base synopsis
    print("\n=== Comparison with Single Base Synopsis ===")
    from spiky_nonconvex import SpikyNonconvexCoordinateDescent
    
    optimizer = SpikyNonconvexCoordinateDescent(
        epsilon=1.0,
        delta=1e-5,
        beta=0.1,
        eta=0.01,
        n=n,
        tau=0.01,
        upper_bound=math.pi,
        lower_bound=-math.pi
    )
    
    # Run with first k queries
    sampled_amplitudes = amplitudes_matrix[:k]
    sampled_frequencies = frequencies_vector[:k]
    sampled_weights = [1.0 / k] * k
    
    optimizer.set_queries_and_amplitudes(sampled_amplitudes, sampled_frequencies, sampled_weights)
    optimizer.real_X = real_data.copy()
    optimizer.generate_data()
    
    single_results = optimizer.run_coordinate_descent(max_iterations=100, verbose=False)
    
    # Compute errors for single synopsis (using optimizer directly)
    single_errors = []
    for query_idx in range(k):
        amps = sampled_amplitudes[query_idx]
        freq = sampled_frequencies[query_idx]
        real_sum = np.sum(real_data**2 + amps * np.sin(freq * real_data))
        real_answer = real_sum / n
        fake_answer = optimizer.fake_output[query_idx]
        error = abs(fake_answer - real_answer)
        single_errors.append(error)
    
    print(f"Single base synopsis statistics (on {k} queries):")
    print(f"  Mean error: {np.mean(single_errors):.6f}")
    print(f"  Median error: {np.median(single_errors):.6f}")
    print(f"  Max error: {np.max(single_errors):.6f}")
    print(f"  Queries with error < 0.5: {np.sum(np.array(single_errors) < 0.5)} / {k}")
    
    return results

if __name__ == "__main__":
    test_boosting()
